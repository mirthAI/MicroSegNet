import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_MicroUS import MicroUS_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='../data/Micro_Ultrasound_Prostate_Segmentation_Dataset/test', help='root dir for test data')
parser.add_argument('--dataset', type=str,
                    default='MicroUS', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists', help='list dir')

parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=10, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_false", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--weight', type=int, default=4, help='weight for hard regions, default is 4')
args = parser.parse_args()


def inference(args, model, test_save_path):
    db_test = MicroUS_dataset(base_dir=args.data_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    result_list = []

    array = np.empty((len(db_test) + 2,3), dtype='U50')
    array[0,1] = "Dice"
    array[0,2] = "HD95"

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        spacing, origin, direction = sampled_batch['spacing'], sampled_batch['origin'], sampled_batch['direction']
        spacing =  [tensor.item() for tensor in spacing]
        origin = [tensor.item() for tensor in origin]
        direction = [tensor.item() for tensor in direction]

        metric_i = test_single_volume(image, label, model, spacing, origin, direction, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name)
        metric_list += np.array(metric_i)
        result_list.append(metric_i[0][0])
        result_list.append(metric_i[0][1])

        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

        array[i_batch+1,0] = case_name
        array[i_batch+1,1] = metric_i[0][0]
        array[i_batch+1,2] = metric_i[0][1]

    metric_list = metric_list / len(db_test)

    mean_dice = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Mean testing performance: mean_dice : %f mean_hd95 : %f ' % (mean_dice, mean_hd95))
    logging.info(result_list)

    array[-1, 0] = "Average"
    array[-1, 1] = mean_dice
    array[-1, 2] = mean_hd95

    # Save csv file
    log_folder = './test_log/test_log_' + args.exp
    save_path = log_folder + '/' + 'test_result.csv'
    np.savetxt(save_path, array, delimiter=",", fmt='%s')

    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'MicroSegNet_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}".format(args.exp)

    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_weight' + str(args.weight)
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 15 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'epoch_'+str(args.max_epochs-1)+'.pth')
    print('The testing model is load from:', snapshot)

    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]


    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)

    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii==1:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, net, test_save_path)



