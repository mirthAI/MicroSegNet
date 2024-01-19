import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer_MicroUS import trainer_MicroUS
from trainer_MicroUS import trainer_MicroUS


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/train_png', help='root dir for training data')
parser.add_argument('--dataset', type=str,
                    default='MicroUS', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists', help='list dir')
parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iteration number to train')
parser.add_argument('--max_epochs', type=int,
                    default=10, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input image size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--weight', type=int,
                    default=4, help='weight for hard regions, default is 4')                
args = parser.parse_args()


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
    args.exp = 'MicroSegNet_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}".format(args.exp)
   
    # Vit name
    snapshot_path += '_' + args.vit_name
    # hard weight
    snapshot_path = snapshot_path + '_weight' + str(args.weight)
    # max epoch
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs)
    # batch size
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    # base learning rate
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    # load model
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {'MicroUS': trainer_MicroUS}
    trainer[dataset_name](args, net, snapshot_path)