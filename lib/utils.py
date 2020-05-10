from lib.constants import IS_GPU, no_of_classes
from torch.utils import data
import torch
import torch.nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, average_precision_score
import seaborn as sns
import sys


def recursive_iterate_modules(m: torch.nn.Module):
    for c in m.children():
        yield c
        yield from recursive_iterate_modules(c)


def segmentation_eval(ious, counts, classes, plot_file_name):
    """
    @param    classes           string          class names
    @param    plot_file_name    string          plot file names
    """
    plot_results(counts, ious, classes, plot_file_name)
    for i in range(len(classes)):
        print('{:>20s}: IoU: {:0.2f}'.format(classes[i], ious[i]))
    print('{:>20s}: IoU: {:0.2f}'.format('mean', np.mean(ious)))
    return ious


def plot_results(counts, ious, classes, file_name):
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    conf = counts / np.sum(counts, 1, keepdims=True)
    conf = np.concatenate([conf, np.array(ious).reshape(-1,1)], 1)
    conf = conf * 100.
    sns.heatmap(conf, annot=True, ax=ax, fmt='3.0f', annot_kws={"size": 12}) 
    arts = [] 
    # labels, title and ticks
    _ = ax.set_xlabel('Predicted labels')
    arts.append(_)
    _ = ax.set_ylabel('True labels')
    arts.append(_)
    _ = ax.set_title('Confusion Matrix, mIoU: {:5.1f}'.format(np.mean(ious)*100.))
    arts.append(_)
    _ = ax.xaxis.set_ticklabels(classes + ['IoU'], rotation=90)
    arts.append(_)
    _ = ax.yaxis.set_ticklabels(classes, rotation=0)
    arts.append(_)
    fig.savefig(file_name, bbox_inches='tight')


def compute_ap(gts, preds):
    aps = []
    for i in range(preds.shape[1]):
      ap, prec, rec = calc_pr(gts == i, preds[:,i:i+1,:,:])
      aps.append(ap)
    return aps


def calc_pr(gt, out, wt=None):
    gt = gt.astype(np.float64).reshape((-1,1))
    out = out.astype(np.float64).reshape((-1,1))

    tog = np.concatenate([gt, out], axis=1)*1.
    ind = np.argsort(tog[:,1], axis=0)[::-1]
    tog = tog[ind,:]
    cumsumsortgt = np.cumsum(tog[:,0])
    cumsumsortwt = np.cumsum(tog[:,0]-tog[:,0]+1)
    prec = cumsumsortgt / cumsumsortwt
    rec = cumsumsortgt / np.sum(tog[:,0])
    ap = voc_ap(rec, prec)
    return ap, rec, prec


def voc_ap(rec, prec):
    rec = rec.reshape((-1,1))
    prec = prec.reshape((-1,1))
    z = np.zeros((1,1)) 
    o = np.ones((1,1))
    mrec = np.vstack((z, rec, o))
    mpre = np.vstack((z, prec, z))

    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    I = np.where(mrec[1:] != mrec[0:-1])[0]+1
    ap = np.sum((mrec[I] - mrec[I-1])*mpre[I])
    return ap


def compute_confusion_matrix(gts, preds):
    # Format: N, H, W
    preds_cls = np.argmax(preds, axis=1).astype(np.uint8)
    gts = gts[:,0,:,:]
    conf = confusion_matrix(gts.ravel(), preds_cls.ravel(), labels=list(range(no_of_classes)))
    inter = np.diag(conf)
    union = np.sum(conf, 0) + np.sum(conf, 1) - np.diag(conf)
    union = np.maximum(union, 1)
    return inter / union, conf


def predict(dataset, model: torch.nn.Module, criterion: torch.nn.CrossEntropyLoss):
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, 
                                 num_workers=0, drop_last=False)
    model.eval()
    gts, preds = [], []
    running_loss: float = 0.0
    with torch.no_grad():
      for batch in dataloader:
        img, gt = batch
        if IS_GPU:
            images = img.cuda()
            gt = gt.cuda()
        outputs = model(Variable(images))
        loss = criterion(outputs, gt[:, 0, :, :])
        outputs = outputs.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        gts.append(gt[0,:,:,:].astype(np.uint8))
        preds.append(outputs[0,:,:,:].astype(np.float16))
        running_loss += loss.item()
    running_loss /= len(dataloader)
    gts = np.array(gts, copy=False)
    preds = np.array(preds, copy=False)
    return gts, preds, list(dataset.classes), running_loss


def torch_compute_confusion_matrix(gts, preds):
    gts = torch.stack(gts, dim=0)
    preds = torch.stack(preds, dim=0)
    batch_confusion = torch.zeros((no_of_classes, no_of_classes), dtype=torch.int64)
    num_classes = no_of_classes

    def check_shape(y, y_pred):
        if y_pred.ndimension() < 2:
            raise ValueError(
                "y_pred must have shape (batch_size, num_categories, ...), " "but given {}".format(y_pred.shape)
            )

        if y_pred.shape[1] != num_classes:
            raise ValueError(
                "y_pred ({}) does not have correct number of categories: {} vs {}".format(
                    y_pred.shape, y_pred.shape[1], num_classes)
            )

        if not (y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError(
                "y_pred must have shape (batch_size, num_categories, ...) and y must have "
                "shape of (batch_size, ...), "
                "but given {} vs {}.".format(y.shape, y_pred.shape)
            )

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if y_shape != y_pred_shape:
            raise ValueError("y and y_pred must have compatible shapes.")

    check_shape(gts, preds)

    # target is (batch_size, ...)
    preds = torch.argmax(preds, dim=1).flatten()
    gts = gts.flatten()

    target_mask = (gts >= 0) & (gts < num_classes)
    gts = gts[target_mask]
    preds = preds[target_mask]

    indices = num_classes * gts + preds
    m = torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    batch_confusion += m.to(batch_confusion)
    batch_confusion = batch_confusion.cpu().numpy().astype(np.float64)

    inter = np.diag(batch_confusion)
    union = np.sum(batch_confusion, 0) + np.sum(batch_confusion, 1) - np.diag(batch_confusion)
    union = np.maximum(union, 1)
    return inter / union, batch_confusion


def predict_and_confuse(dataset, model: torch.nn.Module, criterion: torch.nn.CrossEntropyLoss):
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, 
                                 num_workers=0, drop_last=False)
    model.eval()
    gts, preds = [], []
    running_loss: float = 0.0
    iou = np.zeros((no_of_classes, ), dtype=np.float32)
    total_num_ex = 0
    confusion = np.zeros((no_of_classes, no_of_classes), dtype=np.int64)
    with torch.no_grad():
      for i, batch in enumerate(dataloader):
        img, gt = batch
        if IS_GPU:
            images = img.cuda()
            gt = gt.cuda()
        outputs = model(Variable(images))
        loss = criterion(outputs, gt[:, 0, :, :])
        outputs = outputs.cpu().detach()
        gt = gt.cpu().detach()
        gts.append(gt[0, 0, :, :])
        preds.append(outputs[0, :, :, :])
        running_loss += loss.item()

        if i % 32 == 31:
            num_ex = len(gts)
            batch_iou, batch_confusion = torch_compute_confusion_matrix(gts, preds)
            iou = iou + batch_iou * num_ex
            confusion = confusion + batch_confusion
            gts, preds = [], []
            print('.', end='')
            sys.stdout.flush()
            total_num_ex = total_num_ex + num_ex

    running_loss /= len(dataloader)
    num_ex = len(gts)
    batch_iou, batch_confusion = torch_compute_confusion_matrix(gts, preds)
    iou = iou + batch_iou * num_ex
    confusion = confusion + batch_confusion
    gts, preds = [], []
    total_num_ex = total_num_ex + num_ex
    iou = iou / float(total_num_ex)
    print('')
    return iou, confusion, list(dataset.classes), running_loss
