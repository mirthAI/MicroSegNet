import numpy as np
import torch
from medpy import metric
import cv2
import torch.nn as nn
import SimpleITK as sitk
np.bool = np.bool_


# Annotation-guided binary cross entropy loss (AG-BCE)
def attention_BCE_loss(h_W, y_true, y_pred, y_std, ks = 5):
    number_of_pixels = y_true.shape[0]*y_true.shape[1]*y_true.shape[2]

    y_true_np = y_true.cpu().detach().numpy()
    y_std_np = y_std.cpu().detach().numpy()


    hard = cv2.bitwise_xor(y_true_np, y_std_np)
    hard = hard.astype(np.uint8)
    
    # Apply dilation operation to hard regions
    kernel_size = ks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    for i in range(hard.shape[0]):
        hard[i] = cv2.dilate(hard[i], kernel)
    hard = hard.astype(np.float32)

    easy = abs(hard-1)
    hard = torch.tensor(hard).cuda()
    easy = torch.tensor(easy).cuda()

    epsilon = 0.000001
    beta = 0.5

    loss = -beta*torch.mul(y_true,torch.log(y_pred + epsilon)) - (1.0 - beta)*torch.mul(1.0-y_true,torch.log(1.0 - y_pred + epsilon))
    hard_loss = torch.sum(torch.mul(loss,hard))
    easy_loss = torch.sum(torch.mul(loss,easy))

    LOSS = ((1/(1+h_W))*easy_loss + (h_W/(1+h_W))*hard_loss)/(number_of_pixels)

    return LOSS


def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    hd95 = 0
    dice = 0
    num = 0

    for i in range(pred.shape[0]):
        pred_sum = pred[i,:,:].sum()
        gt_sum = gt[i,:,:].sum()
        if pred_sum>0 and gt_sum>0:
            num +=1
            dice += metric.binary.dc(pred[i,:,:], gt[i,:,:])
            hd95 += metric.binary.hd95(pred[i,:,:], gt[i,:,:])

    hd95 = (hd95*spacing)/num
    dice = dice/num

    return dice, hd95


def test_single_volume(image, label, net, spacing, origin, direction, classes, patch_size=[256, 256], test_save_path=None, case=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]/254.0

        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = cv2.resize(slice, patch_size, interpolation = cv2.INTER_NEAREST)

        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs, _, _, _  = net(input)
            out = torch.sigmoid(outputs).squeeze()

            out = out.cpu().detach().numpy()

            if x != patch_size[0] or y != patch_size[1]:
                pred = cv2.resize(out, (y, x), interpolation = cv2.INTER_NEAREST)
            else:
                pred = out

            a = 1.0*(pred>0.5)
            prediction[ind] = a.astype(np.uint8)

    metric_list = []

    vol_pred = sitk.GetImageFromArray(prediction.astype(np.float32))
    vol_label = sitk.GetImageFromArray(label.astype(np.float32))
    vol_image = sitk.GetImageFromArray(image.astype(np.float32))
    vol_pred.SetSpacing(spacing)
    vol_pred.SetOrigin(origin)
    vol_pred.SetDirection(direction)
    vol_label.SetSpacing(spacing)
    vol_label.SetOrigin(origin)
    vol_label.SetDirection(direction)
    vol_image.SetSpacing(spacing)
    vol_image.SetOrigin(origin)
    vol_image.SetDirection(direction)

    # In-plane spacing is 0.033586mm*0.033586mm
    if classes == 1:
        metric_list.append(calculate_metric_percase(prediction == 1, label == 1, 0.033586))

    else:
        for i in range(1, classes):
            metric_list.append(calculate_metric_percase(prediction == i, label == i, 0.033586))

    if test_save_path is not None:
        sitk.WriteImage(vol_pred, test_save_path + '/'+case + "_pt_label.nii.gz")
        sitk.WriteImage(vol_image, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(vol_label, test_save_path + '/'+ case + "_gt_label.nii.gz")
        

    return metric_list