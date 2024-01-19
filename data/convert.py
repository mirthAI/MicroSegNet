import SimpleITK as sitk
import glob
import cv2


input_image_path = '../Micro_Ultrasound_Prostate_Segmentation_Dataset'
out_image_path = "../train_png/"


list_of_volume = glob.glob(input_image_path + "*MicroUS_downsampled.nii.gz")
list_of_label = glob.glob(input_image_path + "*gd_downsampled.nii.gz")
list_of_st = glob.glob(input_image_path + "*st_downsampled.nii.gz")
list_of_volume = sorted(list_of_volume)
list_of_label = sorted(list_of_label)
list_of_st = sorted(list_of_st)
print(len(list_of_volume))
print(len(list_of_label))
print(len(list_of_st))


down = 2
width = int(1372/down)
height = int(962/down)

for i in range(len(list_of_volume)):  
    img_name = list_of_volume[i]
    gt_name = list_of_label[i]
    st_name = list_of_st[i]
    print(i)
    print(img_name)
    print(gt_name)
    print(st_name)

    img = sitk.ReadImage(img_name)
    gt = sitk.ReadImage(gt_name) 
    st = sitk.ReadImage(st_name)
    image_array = sitk.GetArrayFromImage(img)
    seg_array = sitk.GetArrayFromImage(gt)
    student_seg_array = sitk.GetArrayFromImage(st)
    
    image_array = 255*(image_array - 0)/254
    
    number_of_slices = image_array.shape[0]
    
    for z in range(number_of_slices):
        image_2d = image_array[z]
        if len(seg_array.shape)==3:
            seg_2d = seg_array[z]
            student_seg_2d = student_seg_array[z]

        ### image resizing
        image_2d_resized = cv2.resize(image_2d, (width,height))
        seg_2d_resized = 255*(cv2.resize(seg_2d, (width,height))>0)
        student_seg_2d_resized = 255*(cv2.resize(student_seg_2d, (width,height))>0)

        ### save image to disk
        sub_name = img_name.split("/")[-1].replace("_MicroUS_downsampled.nii.gz","")
        output_image_name = out_image_path + sub_name + "_img_slice_" + str(z) + ".png"
        output_seg_name = out_image_path + sub_name + "_gt_slice_" + str(z) + ".png"
        output_student_seg_name = out_image_path + sub_name + "_st_slice_" + str(z) + ".png"
        
        cv2.imwrite(output_image_name, image_2d_resized)
        cv2.imwrite(output_seg_name, seg_2d_resized)
        cv2.imwrite(output_student_seg_name, student_seg_2d_resized)