import SimpleITK as sitk
import glob
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm     


# Data preprocessing and 2d images generation
image_path = 'Micro_Ultrasound_Prostate_Segmentation_Dataset/train/micro_ultrasound_scans/'
mask_path = 'Micro_Ultrasound_Prostate_Segmentation_Dataset/train/expert_annotations/'
non_exp_path = 'Micro_Ultrasound_Prostate_Segmentation_Dataset/train/non_expert_annotations/'
list_path = '../TransUNet/lists/'
out_image_path = 'train_png/'
test_image_path = 'Micro_Ultrasound_Prostate_Segmentation_Dataset/test/micro_ultrasound_scans/'
test_mask_path = 'Micro_Ultrasound_Prostate_Segmentation_Dataset/test/expert_annotations/'

os.makedirs(out_image_path, exist_ok=True)
os.makedirs(list_path, exist_ok=True)

list_of_image = glob.glob(image_path + "*.nii.gz")
list_of_mask = glob.glob(mask_path + "*.nii.gz")
list_of_st = glob.glob(non_exp_path + "*.nii.gz")
list_of_test_image = glob.glob(test_image_path + "*.nii.gz")
list_of_test_mask = glob.glob(test_mask_path + "*.nii.gz")
list_of_image = sorted(list_of_image)
list_of_mask = sorted(list_of_mask)
list_of_st = sorted(list_of_st)
list_of_test_image = sorted(list_of_test_image)
list_of_test_mask = sorted(list_of_test_mask)

assert len(list_of_image) == len(list_of_mask) == len(list_of_st), 'Each training case must contain image, mask and non-expert annotation.'
assert len(list_of_test_image) == len(list_of_test_mask), 'Each testing case must contain image and mask.'

# Downsample images by 2 for storage first. All images will be resized to 224*224 afterward.
down = 2
width = int(1372/down)
height = int(962/down)

print('Preprocessing starts!')
print('There are {} images, {} masks and {} non-expert annotations for training.'.format(len(list_of_image), len(list_of_mask), len(list_of_st)))
print('There are {} images, {} masks for testing.'.format(len(list_of_test_image), len(list_of_test_mask)))
for i in tqdm(range(len(list_of_image))):  
    img_name = list_of_image[i]
    gt_name = list_of_mask[i]
    st_name = list_of_st[i]

    img = sitk.ReadImage(img_name)
    gt = sitk.ReadImage(gt_name) 
    st = sitk.ReadImage(st_name)
    image_array = sitk.GetArrayFromImage(img)
    seg_array = sitk.GetArrayFromImage(gt)
    student_seg_array = sitk.GetArrayFromImage(st)
    
    image_array = 255*(image_array - 0)/254   # data volumes are normalized to 0-254 beforehand.
    
    number_of_slices = image_array.shape[0]
    
    for z in range(number_of_slices):
        image_2d = image_array[z]
        if len(seg_array.shape)==3:
            seg_2d = seg_array[z]
            student_seg_2d = student_seg_array[z]

        image_2d_resized = cv2.resize(image_2d, (width,height))
        seg_2d_resized = 255*(cv2.resize(seg_2d, (width,height))>0)
        student_seg_2d_resized = 255*(cv2.resize(student_seg_2d, (width,height))>0)

        sub_name = img_name.split("/")[-1].split("_")[0]
        idx = img_name.split("/")[-1].split("_")[-1].split(".")[0]
        output_image_name = out_image_path + sub_name + '_' + idx + "_img_slice_" + str(z) + ".png"
        output_seg_name = out_image_path + sub_name + '_' + idx + "_gt_slice_" + str(z) + ".png"
        output_student_seg_name = out_image_path + sub_name + '_' + idx + "_st_slice_" + str(z) + ".png"
        
        cv2.imwrite(output_image_name, image_2d_resized)
        cv2.imwrite(output_seg_name, seg_2d_resized)
        cv2.imwrite(output_student_seg_name, student_seg_2d_resized)



# Generate CSV file for checking data.
print('Start to generate csv file!')
image_names = sorted(glob.glob(out_image_path + "*img_slice*"))
seg_names = sorted(glob.glob(out_image_path + "*gt_slice*"))
student_seg_names = sorted(glob.glob(out_image_path + "*st_slice*"))

array = np.empty((len(image_names) + 1,5), dtype='U30')
array[0,0] = "image"
array[0,1] = "mask"
array[0,2] = "non_expert_mask"
array[0,3] = "test_img"
array[0,4] = "test_mask"

for i in range(1,len(image_names)+1):
    array[i,0] = image_names[i-1].replace(out_image_path,"").split('.')[0]
    array[i,1] = seg_names[i-1].replace(out_image_path,"").split('.')[0]
    array[i,2] = student_seg_names[i-1].replace(out_image_path,"").split('.')[0]

for i in range(1, len(list_of_test_image)+1):
    array[i,3] = list_of_test_image[i-1].split('/')[-1].split('.')[0]
    array[i,4] = list_of_test_mask[i-1].split('/')[-1].split('.')[0]
    
np.savetxt('data.csv', array, delimiter=",", fmt='%s')
print('Finished generating data.csv file!')



# Generate lists for loading data
print('Start to generate list files!')

# image txt
data= pd.read_csv('data.csv')
key='image'
num = data[key].values.size
name = []
for i in range(num):
    a = data[key].values[i]
    name.append(a)

with open('../TransUNet/lists/image.txt', 'w') as f:
    for item in name:
        f.write("%s\n" % item)

# mask txt
key='mask'
num = data[key].values.size
name = []
for i in range(num):
    a = data[key].values[i]
    name.append(a)

with open('../TransUNet/lists/mask.txt', 'w') as f:
    for item in name:
        f.write("%s\n" % item)

# non-expert txt
key='non_expert_mask'
num = data[key].values.size
name = []
for i in range(num):
    a = data[key].values[i]
    name.append(a)

with open('../TransUNet/lists/non_expert.txt', 'w') as f:
    for item in name:
        f.write("%s\n" % item)

# test image txt
key='test_img'
num = data[key].count()
name = []
for i in range(num):
    a = data[key].values[i]
    name.append(a)

with open('../TransUNet/lists/test_image.txt', 'w') as f:
    for item in name:
        f.write("%s\n" % item)

# test mask txt
key='test_mask'
num = data[key].count()
name = []
for i in range(num):
    a = data[key].values[i]
    name.append(a)

with open('../TransUNet/lists/test_mask.txt', 'w') as f:
    for item in name:
        f.write("%s\n" % item)

print('Finished generating list files!')
print('Preprocessing done!')