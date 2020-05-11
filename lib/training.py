from lib.constants import IS_GPU
from lib.utils import predict_and_confuse, segmentation_eval, compute_confusion_matrix, compute_ap
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from lib.data import SegmentationDataset
import lib.constants
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from sync_bn_lib.sync_batchnorm import DataParallelWithCallback
from typing import Optional
import albumentations
import cv2


EVAL_EVERY = 7


def get_transforms():
    mean_channels = torch.from_numpy(
        np.load('./data/mean_channels.npy').astype(np.float32))
    std_channels = torch.from_numpy(
        np.load('./data/std_channels.npy').astype(np.float32))

    shared_transform = albumentations.Compose([
        albumentations.ShiftScaleRotate(
            shift_limit=0.0425,
            scale_limit=(0.0, 0.1),
            rotate_limit=3,
            p=.75,
            border_mode=cv2.BORDER_CONSTANT,
            value=mean_channels.tolist(),
            # Last class is reserved for misc
            mask_value=lib.constants.no_of_classes - 1),
    ])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_channels / 256, std_channels / 256),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_channels / 256, std_channels / 256)
    ])

    return shared_transform, train_transform, test_transform


class WithLoss(nn.Module):
    def __init__(self, model, loss):
        super(WithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, inputs, targets):
        outputs = self.model(inputs)
        return torch.unsqueeze(self.loss(outputs, targets), dim=-1)


def run_train(model_name: str, model_f, EPOCHS: int, log_dir: str):
    model = model_f()

    class_weights = torch.from_numpy(
        np.load('./data/class_weights.npy').astype(np.float32))
    if IS_GPU:
        model = model.cuda()
        class_weights = class_weights.cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if IS_GPU:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            inner = WithLoss(model, criterion).cuda()
            model = inner.model
            full_model = DataParallelWithCallback(inner)
    print(f'Training model {model_name}')

    shared_transform, train_transform, test_transform = get_transforms()

    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     factor=0.2,
    #     patience=5,
    #     cooldown=5,
    #     verbose=True)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.2)

    train_loss_over_epochs = []
    val_loss_over_epochs = []
    mIoU_over_epochs = []
    mIoU = 0

    best_model: Optional[torch.nn.Module] = None
    best_loss = None

    from tqdm import tqdm
    from tqdm import trange
    import copy

    with trange(EPOCHS, position=0) as tt:
        loss_blurb = None
        best_blurb = None

        for epoch in tt:  # loop over the dataset multiple times
            running_loss = 0.0
            train_dataset = SegmentationDataset(
                split='train', transforms=train_transform, shared_transform=shared_transform)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                                           shuffle=True, num_workers=6,
                                                           drop_last=True)
            val_dataset = SegmentationDataset(
                split='val', transforms=test_transform)
            valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                    num_workers=0, drop_last=False)
            model.train()
            for i, batch in tqdm(enumerate(train_dataloader), position=1, total=len(train_dataloader)):
                # get the inputs
                img, gt = batch

                if IS_GPU:
                    img = img.cuda()
                    gt = gt.cuda()

                optimizer.zero_grad()
                loss = torch.mean(full_model(img, gt[:, 0, :, :]))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                tt.set_postfix(
                    epoch=epoch, loss=loss_blurb, running_loss=running_loss / float(i + 1), best=best_blurb)

            # Normalizing the loss by the total number of train batches
            running_loss /= len(train_dataloader)
            lr_scheduler.step()
            # print('[%d] loss: %.3f' %
            #    (epoch + 1, running_loss))

            train_loss_over_epochs.append(running_loss)
            # Scale of 0.0 to 100.0
            # Calculate validation set accuracy of the existing model
            if epoch % EVAL_EVERY == 0:
                print(
                    f"Evaluating model at epoch {epoch}, training loss was {running_loss}")
                model.eval()
                ious, counts, classes, val_loss = predict_and_confuse(
                    val_dataset, model, criterion)

                import gc
                gc.collect()

                for i in range(len(classes)):
                    print('{:>20s}: IoU: {:0.2f}'.format(classes[i], ious[i]))
                print('{:>20s}: IoU: {:0.2f}'.format('mean', np.mean(ious)))
                print('Validation Loss: {:0.4f}'.format(val_loss))
                mIoU = np.mean(ious)
                mIoU_over_epochs.append(mIoU)
                val_loss_over_epochs.append(val_loss)

                loss_blurb = f'{running_loss}'
                if best_loss is None or val_loss < best_loss:
                    best_loss = val_loss
                    best_model = copy.deepcopy(model).cpu()
                    best_blurb = "Saving Best Model (val loss {:0.4f})".format(
                        val_loss)
                    tt.set_postfix(epoch=epoch, loss=loss_blurb,
                                   best=best_blurb)
                else:
                    best_blurb = None
                    tt.set_postfix(epoch=epoch, loss=running_loss)
            else:
                tt.set_postfix(epoch=epoch, loss=running_loss)

    ########################################################################
    # TODO: Evaluate your result, and report Mean average precision on test dataset
    # using provided helper function. Here we show how we can train and evaluate the
    # simple model that we provided on the validation set. You will want to report
    # performance on the validation set for the variants you tried, and the
    # performance of the final model on the test set.
    assert best_model is not None
    print(f"Evaluting best model ({best_loss})")
    if IS_GPU:
        best_model = best_model.cuda()
    best_model.eval()
    ious, counts, classes, val_loss = predict_and_confuse(
        val_dataset, best_model, criterion)
    ious = segmentation_eval(
        ious, counts, classes, f'{log_dir}/val-{model_name}.pdf')
    print('Final Validation Loss: {:0.4f}'.format(val_loss))

    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator)
    plt.ioff()
    fig = plt.figure(figsize=(16, 12), dpi=192)

    plt.subplot(3, 1, 1)
    plt.ylabel('Train loss')
    plt.plot(np.arange(EPOCHS), train_loss_over_epochs, 'k-')
    plt.title('train loss, val loss, mIoU')
    plt.xticks(np.arange(0, EPOCHS, EVAL_EVERY, dtype=int))
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.ylabel('Validation loss')
    plt.plot(np.arange(0, EPOCHS, EVAL_EVERY), val_loss_over_epochs, 'k-')
    plt.xticks(np.arange(0, EPOCHS, EVAL_EVERY, dtype=int))
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(np.arange(0, EPOCHS, EVAL_EVERY), mIoU_over_epochs, 'r-')
    plt.ylabel('mIoU (val)')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(0, EPOCHS, EVAL_EVERY, dtype=int))
    plt.grid(True)

    plt.savefig(f"{log_dir}/plot_hd_{model_name}.png")
    plt.close(fig)
    print(f'Finished Training {model_name}')
