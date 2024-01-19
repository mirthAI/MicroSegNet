import os
import numpy as np
import torch
import cv2
import SimpleITK as sitk
from torch.utils.data import Dataset


class MultiscaleGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, non_expert = sample['image'], sample['label'], sample['non_expert']

        x, y = image.shape
        image = cv2.resize(image, self.output_size[3])

        label = cv2.resize(label, self.output_size[3])
        label0 = cv2.resize(label, self.output_size[0])
        label1 = cv2.resize(label, self.output_size[1])
        label2 = cv2.resize(label, self.output_size[2])

        non_expert = cv2.resize(non_expert, self.output_size[3])
        non_expert0 = cv2.resize(non_expert, self.output_size[0])
        non_expert1 = cv2.resize(non_expert, self.output_size[1])
        non_expert2 = cv2.resize(non_expert, self.output_size[2])

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

        label = torch.from_numpy(label.astype(np.float32))
        label0 = torch.from_numpy(label0.astype(np.float32))
        label1 = torch.from_numpy(label1.astype(np.float32))
        label2 = torch.from_numpy(label2.astype(np.float32))

        non_expert = torch.from_numpy(non_expert.astype(np.float32))
        non_expert0 = torch.from_numpy(non_expert0.astype(np.float32))
        non_expert1 = torch.from_numpy(non_expert1.astype(np.float32))
        non_expert2 = torch.from_numpy(non_expert2.astype(np.float32))

        sample = {'image': image, 'label': label.long(), 'label0': label0.long(), 'label1': label1.long(), 'label2': label2.long(), 
                  'non_expert': non_expert.long(), 'non_expert0': non_expert0.long(), 'non_expert1': non_expert1.long(), 'non_expert2': non_expert2.long()}
        return sample


class MicroUS_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.data_dir = base_dir
        self.image_list = open(os.path.join(list_dir, 'image'+'.txt')).readlines()
        self.label_list = open(os.path.join(list_dir, 'mask'+'.txt')).readlines()
        self.non_expert_list = open(os.path.join(list_dir, 'non_expert'+'.txt')).readlines()
        self.test_image_list = open(os.path.join(list_dir, 'test'+'_image'+'.txt')).readlines()
        self.test_label_list = open(os.path.join(list_dir, 'test'+'_mask'+'.txt')).readlines()

        
    def __len__(self):
        return len(self.image_list) if self.split == 'train' else len(self.test_image_list)

    def __getitem__(self, idx):
        if self.split == "train":
            image_name = self.image_list[idx].strip('\n')
            label_name = self.label_list[idx].strip('\n')
            non_expert_name = self.non_expert_list[idx].strip('\n')
            image_path = os.path.join(self.data_dir, image_name+'.png')
            label_path = os.path.join(self.data_dir, label_name+'.png')
            non_expert_path = os.path.join(self.data_dir, non_expert_name+'.png')
            image = cv2.imread(image_path)[:,:,0]/255.0
            label = cv2.imread(label_path)[:,:,0]/255.0
            non_expert = cv2.imread(non_expert_path)[:,:,0]/255.0

            sample = {'image': image, 'label': label, 'non_expert': non_expert}
            sample['case_name'] = self.image_list[idx].strip('\n')

        else:
            vol_image_name = self.test_image_list[idx].strip('\n')
            filepath = self.data_dir + '/micro_ultrasound_scans' + "/{}.nii.gz".format(vol_image_name)
            vol_image = sitk.ReadImage(filepath)
            spacing = vol_image.GetSpacing()
            origin = vol_image.GetOrigin()
            direction = vol_image.GetDirection()
            vol_image = sitk.GetArrayFromImage(vol_image)

            vol_label_name = self.test_label_list[idx].strip('\n')
            filepath = self.data_dir + '/expert_annotations' + "/{}.nii.gz".format(vol_label_name)
            vol_label = sitk.ReadImage(filepath)
            vol_label = sitk.GetArrayFromImage(vol_label)

            image, label = vol_image, vol_label

            sample = {'image': image, 'label': label}
            sample['case_name'] = self.test_image_list[idx].strip('\n')
            sample['spacing'] = spacing
            sample['origin'] = origin
            sample['direction'] = direction

        if self.transform:
            sample = self.transform(sample)

        
        return sample