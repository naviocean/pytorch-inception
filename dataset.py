from PIL import Image
import os
import os.path
import glob
import numpy as np
from torch.utils.data import Dataset
import random


class MyDataset(Dataset):
    def __init__(self, root_path, data_folder='train', name_list='ucfTrainTestlist', version=1, transform=None, num_frames=3):
        self.root_path = root_path
        self.data_folder = data_folder
        self.num_frames = num_frames
        self.random = random
        self.split_file = os.path.join(self.root_path, name_list,
                                       str(data_folder) + 'list0' + str(version) + '.txt')
        self.label_file = os.path.join(self.root_path, name_list, 'classInd.txt')
        self.label_dict = self.get_labels()

        self.images_dict = self.get_images_list()
        # self.video_dict = self.get_video_list()

        self.version = version
        self.transform = transform

        # self.__getitem__(1)

    # def get_video_list(self):
    #     res = []
    #     with open(self.split_file) as fin:
    #         for line in list(fin):
    #             line = line.replace("\n", "")
    #             split = line.split(" ")
    #             # get number frames of each video
    #             video_path = split[0].split('.')[0]
    #             frames_path = os.path.join(self.root_path, self.data_folder, video_path)
    #             allfiles = glob.glob(frames_path + '/*.jpg')
    #             # remove video which has < 16 image frames
    #             if len(allfiles) >= 16:
    #                 res.append(split[0])
    #     return res

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
                allfiles = sorted(allfiles)
                total = len(allfiles) - 1;
                files = []
                if total >= 30:
                    num_div = total // self.num_frames
                    # print(total, num_div)
                    for j in range(self.num_frames):
                        end = (j+1)*num_div
                        if end > total:
                            end = total
                        l_range = list(range(j*num_div, end))
                        # idx_image = random.sample(l_range, 1)
                        idx_image = l_range[0]
                        files.append(allfiles[idx_image])
                        # print(l_range, idx_image, allfiles[idx_image])
                if len(files) > 0:
                    res = res + files
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

    # def get_random_image(self, dir, file_ext="jpg", sort_files=True):
    #     allfiles = glob.glob(dir + '/*.' + file_ext)
    #     allfiles = sorted(allfiles)
    #     image = random.sample(allfiles, 1)
    #     return image[0]
    #
    # def get_video_tensor(self, dir):
    #     image = self.get_random_image(dir)
    #     image = Image.open(image)
    #     if self.transform is not None:
    #         image = self.transform(image);
    #     return image

    # def __getitem__(self, index):
    #     video = self.video_dict[index]
    #     video_path = video.split('.')[0]
    #
    #     frames_path = os.path.join(self.root_path, self.data_folder, video_path)
    #     clip = self.get_video_tensor(frames_path)
    #     # get label name from video path
    #     label_name = video_path.split('/')[0]
    #     label_index = self.label_dict[label_name];
    #     return (clip, label_index)
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
