import argparse
import cv2
import pickle
import os
import numpy as np
from collections import namedtuple
import random
from lib.cityscape_labels import labels, Label
from pathlib import Path
from tqdm import trange, tqdm
from lib.constants import new_img_height, new_img_width, no_of_classes


def dir_path(string):
    if os.path.isdir(string):
        return Path(string)
    else:
        raise NotADirectoryError(string)


parser = argparse.ArgumentParser()
parser.add_argument("--project-dir", type=dir_path,
                    required=True, help="project directory")
parser.add_argument("--data-dir", type=dir_path,
                    required=True, help="raw data directory")

args = parser.parse_args()

project_dir: Path = args.project_dir
data_dir: Path = args.data_dir


# create a function mapping id to trainId:
id_to_trainId = {label.id: label.trainId for label in labels}
id_to_trainId_map_func = np.vectorize(id_to_trainId.get)

# (the height all images fed to the model will be resized to)
new_img_height = 512
# (the width all images fed to the model will be resized to)
new_img_width = 1024
no_of_classes = 20  # (number of object classes (road, sidewalk, car etc.))

cityscapes_dir = data_dir

train_imgs_dir = cityscapes_dir / "leftImg8bit" / "train"
train_gt_dir = cityscapes_dir / "gtFine" / "train"

val_imgs_dir = cityscapes_dir / "leftImg8bit" / "val"
val_gt_dir = cityscapes_dir / "gtFine" / "val"

train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
              "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
              "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
              "bremen/", "bochum/", "aachen/"]
val_dirs = ["frankfurt/", "munster/", "lindau/"]


# get the path to all training images and their corresponding label image:
train_img_paths = []
train_trainId_label_paths = []
for train_dir in tqdm(train_dirs, desc="Training directories", position=0):
    img_dir = train_imgs_dir / train_dir

    file_names = os.listdir(img_dir)
    for file_name in tqdm(file_names, desc="Files", position=1):
        img_id = file_name.split("_left")[0]

        # read the image:
        img_path = img_dir / file_name
        img = cv2.imread(str(img_path), -1)

        # resize the image without interpolation (want the image to still match
        # the corresponding label image which we reisize below) and save to
        # project_dir/data:
        img_small = cv2.resize(img, (new_img_width, new_img_height),
                               interpolation=cv2.INTER_NEAREST)
        img_small_path = project_dir / "data" / "raw" / (img_id + ".png")
        cv2.imwrite(str(img_small_path), img_small)
        train_img_paths.append(img_small_path)

        # read and resize the corresponding label image without interpolation
        # (want the resulting image to still only contain pixel values
        # corresponding to an object class):
        gt_img_path = train_gt_dir / train_dir / (img_id + "_gtFine_labelIds.png")
        gt_img = cv2.imread(str(gt_img_path), -1)
        gt_img_small = cv2.resize(gt_img, (new_img_width, new_img_height),
                                  interpolation=cv2.INTER_NEAREST)

        # convert the label image from id to trainId pixel values:
        id_label = gt_img_small
        trainId_label = id_to_trainId_map_func(id_label)

        # save the label image to project_dir/data:
        trainId_label_path = project_dir / "data" / "raw" / (img_id + "_trainId_label.png")
        cv2.imwrite(str(trainId_label_path), trainId_label)
        train_trainId_label_paths.append(trainId_label_path)


# compute the mean color channels of the train imgs:
print("computing mean color channels of the train imgs")
no_of_train_imgs = len(train_img_paths)
mean_channels = np.zeros((3, ), dtype=np.float32)
for img_path in tqdm(train_img_paths):
    img = np.array(cv2.imread(str(img_path), -1)).astype(np.float32)

    img_mean_channels = np.mean(img, axis=0)
    img_mean_channels = np.mean(img_mean_channels, axis=0)

    mean_channels += img_mean_channels

mean_channels = mean_channels / float(no_of_train_imgs)

# # save to disk:
np.save(project_dir / "data" / "mean_channels.npy", mean_channels)

print("computing std color channels of the train imgs")

no_of_train_imgs = len(train_img_paths)
std_channels = np.zeros((3, ), dtype=np.float32)
for img_path in tqdm(train_img_paths):
    img = np.array(cv2.imread(str(img_path), -1)).astype(np.float32)
    img_sub_mean = img - mean_channels
    val = np.mean(np.mean(np.power(img_sub_mean, 2), axis=0), axis=0)
    std_channels += val

std_channels = np.sqrt(std_channels / float(no_of_train_imgs))

np.save(project_dir / "data" / "std_channels.npy", std_channels)

# compute the class weights:
print("computing class weights")
trainId_to_count = {}
for trainId in range(no_of_classes):
    trainId_to_count[trainId] = 0

# # get the total number of pixels in all train labels that are of each
# # object class:
for trainId_label_path in tqdm(train_trainId_label_paths):
    # read the label image:
    trainId_label = cv2.imread(str(trainId_label_path), -1)

    for trainId in range(no_of_classes):
        # count how many pixels in the label image are of object class trainId:
        trainId_mask = np.equal(trainId_label, trainId)
        label_trainId_count = np.sum(trainId_mask)

        # add to the total count:
        trainId_to_count[trainId] += label_trainId_count

# # compute the class weights according to the paper:
class_weights = []
total_count = sum(trainId_to_count.values())
for trainId, count in list(trainId_to_count.items()):
    trainId_prob = float(count) / float(total_count)
    trainId_weight = 1 / np.log(1.02 + trainId_prob)
    class_weights.append(trainId_weight)

# # save to disk:
np.save(project_dir / "data" / "class_weights.npy", np.array(class_weights))


# get the path to all validation images and their corresponding label image:
val_img_paths = []
val_trainId_label_paths = []
for val_dir in tqdm(val_dirs, desc="Validation directories", position=0):
    img_dir = val_imgs_dir / val_dir

    file_names = os.listdir(img_dir)
    for file_name in tqdm(file_names, desc="Files", position=1):
        img_id = file_name.split("_left")[0]

        # read the image:
        img_path = img_dir / file_name
        img = cv2.imread(str(img_path), -1)

        # resize the image without interpolation (want the image to still match
        # the corresponding label image which we reisize below) and save to
        # project_dir/data:
        img_small = cv2.resize(img, (new_img_width, new_img_height),
                               interpolation=cv2.INTER_NEAREST)
        img_small_path = project_dir / "data" / "raw" / (img_id + ".png")
        cv2.imwrite(str(img_small_path), img_small)
        val_img_paths.append(img_small_path)

        # read and resize the corresponding label image without interpolation
        # (want the resulting image to still only contain pixel values
        # corresponding to an object class):
        gt_img_path = val_gt_dir / val_dir / (img_id + "_gtFine_labelIds.png")
        gt_img = cv2.imread(str(gt_img_path), -1)
        gt_img_small = cv2.resize(gt_img, (new_img_width, new_img_height),
                                  interpolation=cv2.INTER_NEAREST)

        # convert the label image from id to trainId pixel values:
        id_label = gt_img_small
        trainId_label = id_to_trainId_map_func(id_label)

        # save the label image to project_dir/data:
        trainId_label_path = project_dir / "data" / "raw" / (img_id + "_trainId_label.png")
        cv2.imwrite(str(trainId_label_path), trainId_label)
        val_trainId_label_paths.append(trainId_label_path)

# # save the validation data to disk:
pickle.dump(val_trainId_label_paths, open(project_dir / "data" / "val_trainId_label_paths.pkl", "wb"))
pickle.dump(val_img_paths, open(project_dir / "data" / "val_img_paths.pkl", "wb"))


# augment the train data by flipping all train imgs:
no_of_train_imgs = len(train_img_paths)
print("number of train imgs before augmentation: %d " % no_of_train_imgs)

augmented_train_img_paths = []
augmented_train_trainId_label_paths = []
for (img_path, label_path) in tqdm(zip(train_img_paths, train_trainId_label_paths), total=len(train_img_paths)):
    augmented_train_img_paths.append(img_path)
    augmented_train_trainId_label_paths.append(label_path)

    # read the image:
    img = cv2.imread(str(img_path), -1)

    # flip the image and save to project_dir/data:
    img_flipped = cv2.flip(img, 1)
    img_flipped_path = str(img_path).split(".png")[0] + "_flipped.png"
    cv2.imwrite(img_flipped_path, img_flipped)
    augmented_train_img_paths.append(img_flipped_path)

    # read the corresponding label image:
    label_img = cv2.imread(str(label_path), -1)

    # flip the label image and save to project_dir/data:
    label_img_flipped = cv2.flip(label_img, 1)
    label_img_flipped_path = str(label_path).split(".png")[0] + "_flipped.png"
    cv2.imwrite(label_img_flipped_path, label_img_flipped)
    augmented_train_trainId_label_paths.append(label_img_flipped_path)

# # randomly shuffle the augmented train data:
augmented_train_data = list(zip(augmented_train_img_paths,
                                augmented_train_trainId_label_paths))
random.shuffle(augmented_train_data)
random.shuffle(augmented_train_data)
random.shuffle(augmented_train_data)
random.shuffle(augmented_train_data)

# # save the augmented train data to disk:
train_data = augmented_train_data
train_img_paths, train_trainId_label_paths = list(zip(*train_data))
pickle.dump(train_img_paths, open(project_dir / "data" / "train_img_paths.pkl", "wb"))
pickle.dump(train_trainId_label_paths, open(project_dir / "data" / "train_trainId_label_paths.pkl", "wb"))

no_of_train_imgs = len(train_img_paths)
print("number of train imgs after augmentation: %d " % no_of_train_imgs)
