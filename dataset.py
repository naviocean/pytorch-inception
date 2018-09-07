from PIL import Image
import os
import os.path
import glob
import numpy as np
from torch.utils.data import Dataset
import random


class MyDataset(Dataset):
    def __init__(self, root_path, data_folder='train', name_list='ucfTrainTestlist', version=1, transform=None):
        self.root_path = root_path
        self.data_folder = data_folder
        self.random = random
        self.split_file = os.path.join(self.root_path, name_list,
                                       str(data_folder) + 'list0' + str(version) + '.txt')
        self.label_file = os.path.join(self.root_path, name_list, 'classInd.txt')
        self.label_dict = self.get_labels()

        self.images_dict = self.get_images_list()

        self.version = version
        self.transform = transform

    def get_images_list(self):
        res = []
        with open(self.split_file) as fin:
            for line in list(fin):
                line = line.replace("\n", "")
                split = line.split(" ")
                # get number frames of each video
                video_path = split[0].split('.')[0]
                frames_path = os.path.join(self.root_path, self.data_folder, video_path)
                allfiles = glob.glob(frames_path + '/*.jpg')
                res = res + allfiles
        return res

    # Get all labels from classInd.txt
    def get_labels(self):
        label_dict = {}
        with open(self.label_file) as fin:
            for row in list(fin):
                row = row.replace("\n", "").split(" ")
                # -1 because the index of array is start from 0
                label_dict[row[1]] = int(row[0]) - 1
        return label_dict

    # stuff
    def __getitem__(self, index):
        image = self.images_dict[index]
        img = Image.open(image)
        root_path = os.path.join(self.root_path, self.data_folder)
        path = image.replace(root_path, '')
        label_name = path.split('/')[1]
        label_index = self.label_dict[label_name]

        if self.transform is not None:
            img = self.transform(img)
        return (img, label_index)

    def __len__(self):
        return len(self.images_dict)


if __name__ == '__main__':
    MyDataset('/Users/naviocean/data/UCF101/', 'validation')
