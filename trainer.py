import os
import argparse
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np
import copy
import shutil
import csv
import queue
import time

from torch.nn import functional as F
from networks.vision_transformer import SwinUnet as ViT_seg

from data_utils.transformer_2d import Get_ROI, RandomFlip2D, RandomRotate2D, RandomErase2D, RandomAdjust2D, \
    RandomDistort2D, RandomZoom2D, RandomNoise2D
from data_utils.data_loader import DataGenerator, CropResize, To_Tensor0, To_Tensor1, DataGeneratorval, \
    Trunc_and_Normalize, RandomGenerator
from strategy.ap_strategy import acc_predictor
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

import random
import warnings

warnings.filterwarnings('ignore')
import setproctitle

from utils import dfs_remove_weight
from torch.nn.modules.loss import CrossEntropyLoss
from config_vit import get_config

# GPU version.
##
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument(
    '--cfg', type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
args = parser.parse_args()
config = get_config(args)
##
import numpy as np
from skimage.segmentation import felzenszwalb, slic
import torch
import torch.nn.functional as F
from skimage.segmentation import watershed
from scipy.ndimage import median_filter
import cv2


class DiceLoss0(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss0, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def label_propagation(image, scribble, dataset='ACDC', method='felzenszwalb'):
    """
    image: BCHWD 0~1 torch.tensor
    scribble: BCHWD torch.tensor
    """

    # img 归一化
    image = (image - image.min()) / (image.max() - image.min()) * 255
    # array
    image = image.cpu().numpy().astype(np.uint8)
    scribble = np.array(scribble)
    if dataset == 'ACDC':
        scribble[scribble == 4] = 0
    elif dataset == 'CHAOS':
        scribble[scribble == 5] = 0
    elif dataset == 'VS':
        scribble[scribble == 2] = 0
    elif dataset == 'RUIJIN':
        scribble[scribble == 2] = 0

    B, C, H, W = image.shape
    pseudo_mask = np.zeros(image.shape)
    su_mask = np.zeros(image.shape)
    for b in range(B):
        # 找前景区域，只在前景区域寻找pseudo label
        x, y = np.where(scribble[b, :, :] != 0)
        if x.size == 0:
            continue
        x_min, x_max, y_min, y_max = max((x.min() - 10), 0), min((x.max() + 10), scribble.shape[1]), \
                                     max((y.min() - 10), 0), min((y.max() + 10), scribble.shape[2])

        img_fg = image[b, 0, x_min:x_max, y_min:y_max]
        scr_fg = scribble[b, x_min:x_max, y_min:y_max]
        H_fg, W_fg = img_fg.shape
        pseudo_fg = np.zeros(img_fg.shape)
        su_fg = np.zeros(img_fg.shape)

        for d in range(1):
            img = img_fg[:, :]
            scr = scr_fg[:, :]
            su = felzenszwalb(img, scale=50, sigma=0.5, min_size=30)

            su_fg[:, :] = su
            scribble_value_list = np.unique(scr)
            scribble_value_ignore = 0
            for scribble_value in scribble_value_list:
                if scribble_value != scribble_value_ignore:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    tmp = scr.copy()
                    tmp[scr == scribble_value] = 1
                    tmp[scr != scribble_value] = 0
                    if dataset == 'ACDC':
                        valid_mask = cv2.dilate(tmp, kernel, iterations=1)
                    if 'CHAOS' in dataset:
                        valid_mask = cv2.dilate(tmp, kernel, iterations=5)
                    if dataset == 'VS':
                        valid_mask = cv2.dilate(tmp, kernel, iterations=1)
                    if dataset == 'RUIJIN':
                        valid_mask = cv2.dilate(tmp, kernel, iterations=1)
                    supervoxel_under_scribble_marking = np.unique(su[scr == scribble_value])
                    tmp_mask = np.zeros(img.shape)
                    for i in supervoxel_under_scribble_marking:
                        tmp_mask[su == i] = scribble_value
                    if dataset != 'VS':
                        tmp_mask *= valid_mask
                    for h in range(H_fg):
                        for w in range(W_fg):
                            if tmp_mask[h, w] != 0:
                                pseudo_fg[h, w] = tmp_mask[h, w]
        pseudo_mask[b, 0, x_min:x_max, y_min:y_max] = pseudo_fg
        su_mask[b, 0, x_min:x_max, y_min:y_max] = su_fg

    return torch.Tensor(pseudo_mask), torch.Tensor(su_mask)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class SemanticSeg(object):
    '''
    Control the training, evaluation, and inference process.
    Args:
    - net_name: string
    - lr: float, learning rate.
    - n_epoch: integer, the epoch number
    - channels: integer, the channel number of the input
    - num_classes: integer, the number of class
    - input_shape: tuple of integer, input dim
    - crop: integer, cropping size
    - batch_size: integer
    - num_workers: integer, how many subprocesses to use for data loading.
    - device: string, use the specified device
    - pre_trained: True or False, default False
    - weight_path: weight path of pre-trained model
    '''

    def __init__(self,
                 net_name=None,
                 encoder_name=None,
                 predictor_name=None,
                 lr=1e-3,
                 n_epoch=100,
                 warmup_epoch=5,
                 sample_inteval=5,
                 channels=1,
                 num_classes=2,
                 target_names=None,
                 max_percent=0.5,
                 init_percent=0.1,
                 roi_number=1,
                 scale=None,
                 input_shape=None,
                 crop=48,
                 batch_size=6,
                 num_workers=0,
                 device=None,
                 pre_trained=False,
                 ex_pre_trained=False,
                 ckpt_point=True,
                 seg_weight_path=None,
                 predictor_weight_path=None,
                 weight_decay=0.,
                 momentum=0.95,
                 gamma=0.1,
                 milestones=[40, 80],
                 T_max=5,
                 mean=None,
                 std=None,
                 topk=50,
                 use_fp16=True):
        super(SemanticSeg, self).__init__()

        self.net_name = net_name
        self.encoder_name = encoder_name
        self.predictor_name = predictor_name

        self.lr = lr
        self.n_epoch = n_epoch
        self.warmup_epoch = warmup_epoch
        self.sample_inteval = sample_inteval
        self.channels = channels
        self.num_classes = num_classes
        self.target_names = target_names
        self.max_percent = max_percent
        self.init_percent = init_percent
        self.roi_number = roi_number
        self.scale = scale
        self.input_shape = input_shape
        self.crop = crop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.pre_trained = pre_trained
        self.ex_pre_trained = ex_pre_trained
        self.ckpt_point = ckpt_point
        self.seg_weight_path = seg_weight_path
        self.predictor_weight_path = predictor_weight_path

        self.start_epoch = 0
        self.global_step = 0
        self.seg_loss_threshold = 1.0
        self.metrics_threshold = 0.

        self.weight_decay = weight_decay
        self.momentum = momentum
        self.gamma = gamma
        self.milestones = milestones
        self.T_max = T_max
        self.mean = mean
        self.std = std
        self.topk = topk
        self.use_fp16 = use_fp16

        os.environ['CUDA_VISIBLE_DEVICES'] = self.device

        self.net = self._get_seg_net(self.net_name)
        self.net2 = self._get_seg_net('vit')
        self.net3 = self._get_seg_net('vit_m')
        self.predictor = self._get_predictor(self.predictor_name)

        if self.pre_trained:
            self._get_pre_trained_seg_net(self.seg_weight_path, ckpt_point)
            self._get_pre_trained_predictor(self.predictor_weight_path, ckpt_point)

        if self.roi_number is not None:
            assert self.num_classes == 2, "num_classes must be set to 2 for binary segmentation"

        self.get_roi = False

    def trainer(self,
                train_path,
                val_path,
                cur_fold,
                output_dir=None,
                log_dir=None,
                optimizer='Adam',
                seg_loss_fun='Cross_Entropy',
                seg_loss_fund='DiceLoss',
                predictor_loss_fun='MSE',
                sample_mode='linear',
                sample_from_all_data=True,
                sample_weight=None,
                al_mode='ap',
                score_type='mean',
                class_weight=None,
                lr_scheduler=None,
                freeze_encoder=False,
                get_roi=False,
                repeat_factor=1.0,
                sample_strategy='norm',
                sample_patience=10,
                sample_times=10,
                args=None):

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        torch.cuda.manual_seed_all(0)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        output_dir = os.path.join(output_dir, "fold" + str(cur_fold))
        log_dir = os.path.join(log_dir, "fold" + str(cur_fold))

        if os.path.exists(log_dir):
            if not self.pre_trained:
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
        else:
            os.makedirs(log_dir)

        if os.path.exists(output_dir):
            if not self.pre_trained:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)

        if not self.pre_trained:
            with open(os.path.join(log_dir, 'record.csv'), 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                rows = ['epoch', 'label_ratio', 'unlabel_ratio', 'train_predictor_loss', 'val_predictor_loss', \
                        'train_run_dice', 'val_run_dice']
                rows.extend(self.target_names * 2)
                csvwriter.writerow(rows)

        def record_sample_list(record_csv, key, sample_list=[]):
            import pandas as pd
            if not os.path.exists(record_csv):
                df = pd.DataFrame(columns=[str(i) for i in range(self.n_epoch)])
                df['col'] = [np.nan] * len(train_path)
                df.to_csv(record_csv, index=False)
            if len(sample_list) != 0:
                df = pd.read_csv(record_csv)
                df[key][:len(sample_list)] = [os.path.basename(case) for case in sample_list]
                df.to_csv(record_csv, index=False)

        segnet_output_dir = os.path.join(output_dir, 'segnet')
        predictor_output_dir = os.path.join(output_dir, 'predictor')

        if not os.path.exists(segnet_output_dir):
            os.makedirs(segnet_output_dir)
        if not os.path.exists(predictor_output_dir):
            os.makedirs(predictor_output_dir)

        # self.step_per_epoch = len(train_path) // self.batch_size
        self.writer = SummaryWriter(log_dir)
        if not self.pre_trained:
            if args is not None and isinstance(args, dict):
                for key, value in args.items():
                    self.writer.add_text(key, str(value), 0)

        net = self.net
        net2 = self.net2
        net3 = self.net3
        predictor = self.predictor

        if freeze_encoder:
            for param in net.encoder.parameters():
                param.requires_grad = False

        lr = self.lr
        loss = self._get_seg_loss(seg_loss_fun, class_weight)
        loss_s = CrossEntropyLoss(ignore_index=4)
        loss_d = self._get_seg_loss(seg_loss_fund, class_weight)
        loss_dd = DiceLoss0(n_classes=4)
        predictor_loss = self._get_predictor_loss(predictor_loss_fun)

        if len(self.device.split(',')) > 1:
            net = DataParallel(net)
            predictor = DataParallel(predictor)

        # copy to gpu
        net = net.cuda()
        predictor = predictor.cuda()
        loss = loss.cuda()
        loss_s = loss_s.cuda()
        predictor_loss = predictor_loss.cuda()

        # optimizer setting
        seg_optimizer = self._get_optimizer(optimizer, net, lr)
        seg_optimizer2 = self._get_optimizer(optimizer, net2, lr)
        predictor_optimizer = self._get_optimizer(optimizer, predictor, lr)

        scaler = GradScaler()
        predictor_scaler = GradScaler()

        if lr_scheduler is not None:
            seg_lr_scheduler = self._get_lr_scheduler(lr_scheduler, seg_optimizer)
            predictor_lr_scheduler = self._get_lr_scheduler(lr_scheduler, predictor_optimizer)

        # loss_threshold = 1.0
        early_stopping = EarlyStopping(patience=40,
                                       verbose=True,
                                       delta=1e-3,
                                       monitor='val_run_dice',
                                       op_type='max')

        # dataloader setting
        data_size = len(train_path)
        self.repeat_factor = repeat_factor
        self.get_roi = get_roi

        if not self.pre_trained:
            self.unlabeled_data_pool = copy.deepcopy(train_path)
            print('unlabeled sample num = %d' % len(self.unlabeled_data_pool))
            self.labeled_data_pool = []
        # sample_times = int((self.n_epoch - self.warmup_epoch)/self.sample_inteval)

        self.sample_times = sample_times

        self.samples_per_epoch = self._get_samples_per_epoch(
            N_sample=len(train_path),
            sample_times=self.sample_times,
            sample_mode=sample_mode)

        print(self.samples_per_epoch)
        print('total sample num = %d' % sum(self.samples_per_epoch))
        self.sample_queue = queue.Queue()
        for item in self.samples_per_epoch[1:]:
            self.sample_queue.put(item)

        val_loader = self._get_data_loader(val_path, 'val', repeat_factor=1.0)
        sample_count = 0
        total_sample_time = 0
        for epoch in range(self.start_epoch, self.n_epoch):

            setproctitle.setproctitle('{}: {}/{}'.format('User', epoch, self.n_epoch))

            sample_flag = False
            labeled_data = []
            if epoch == 0:
                labeled_data = self._random_sampling(sample_pool=train_path)

            else:
                if sample_strategy == 'iq':
                    if early_stopping.counter >= sample_patience and not self.sample_queue.empty():
                        sample_flag = True
                        early_stopping.counter = 0

                else:
                    if epoch >= self.warmup_epoch and ((epoch - self.warmup_epoch) % self.sample_inteval == 0) \
                            and not self.sample_queue.empty():
                        sample_flag = True

            if sample_flag:
                if not sample_from_all_data:
                    unlabeled_data_pool = random.sample(self.unlabeled_data_pool, k=int(
                        len(self.unlabeled_data_pool) * 0.1))  # 0.1 can be set to another value
                else:
                    unlabeled_data_pool = self.unlabeled_data_pool
                sample_loader = self._get_data_loader(unlabeled_data_pool, 'val', repeat_factor=1.0)
                sample_nums = self.sample_queue.get()
                if sample_nums != 0:
                    sample_count += 1
                    s_time = time.time()
                    print(
                        f'************* start sampling : {sample_nums}, count:{sample_count}/{self.sample_times} *************')
                    labeled_data = acc_predictor(
                        seg_net=net,
                        predictor=predictor,
                        unlabeled_data_pool=unlabeled_data_pool,
                        sample_loader=sample_loader,
                        sample_nums=sample_nums,
                        sample_weight=sample_weight,
                        al_mode=al_mode,
                        score_type=score_type)
                    print(f'************* finish sampling : {sample_nums} *************')
                    total_sample_time += time.time() - s_time
                    print('sample time:%.4f' % (time.time() - s_time))

            if len(labeled_data) != 0:
                self.labeled_data_pool.extend(labeled_data)
                self._update_unlabeled_data_pool(labeled_data=labeled_data)
                random.shuffle(self.labeled_data_pool)
                train_loader = self._get_data_loader(self.labeled_data_pool, 'train', repeat_factor=self.repeat_factor)

            if epoch < 5:
                sample_loader1 = train_loader
            else:
                if not sample_from_all_data:
                    unlabeled_data_pool = random.sample(self.unlabeled_data_pool, k=int(
                        len(self.unlabeled_data_pool) * 0.1))  # 0.1 can be set to another value
                else:
                    unlabeled_data_pool = self.unlabeled_data_pool
                sample_loader = self._get_data_loader(unlabeled_data_pool, 'val', repeat_factor=1.0)
                sample_loader1 = sample_loader

            train_loss, train_dice, train_run_dice, train_predictor_loss = self._train_on_epoch(
                epoch=epoch,
                net=net,
                net2=net2,
                net3=net3,
                predictor=predictor,
                criterion=loss_s,
                criteriond=loss_d,
                criteriondd=loss_dd,
                predictor_criterion=predictor_loss,
                optimizer=seg_optimizer,
                optimizer2=seg_optimizer2,
                predictor_optimizer=predictor_optimizer,
                scaler=scaler,
                predictor_scaler=predictor_scaler,
                train_loader=train_loader,
                sample_loader=sample_loader1,
            )

            val_loss, val_dice, val_run_dice, val_predictor_loss = self._val_on_epoch(
                epoch=epoch,
                net=net,
                predictor=predictor,
                criterion=loss,
                predictor_criterion=predictor_loss,
                val_loader=val_loader)

            if seg_lr_scheduler is not None:
                seg_lr_scheduler.step()

            if predictor_lr_scheduler is not None and epoch >= (self.warmup_epoch - self.sample_inteval):
                predictor_lr_scheduler.step()

            torch.cuda.empty_cache()

            print(
                'epoch:{},train_loss:{:.5f},train_predictor_loss:{:.5f},val_loss:{:.5f},val_predictor_loss:{:.5f}'.format(
                    epoch, train_loss, train_predictor_loss, val_loss, val_predictor_loss))

            print('epoch:{},train_dice:{:.5f},train_run_dice:{:.5f},val_dice:{:.5f},val_run_dice:{:.5f}'
                  .format(epoch, train_dice, train_run_dice[0], val_dice, val_run_dice[0]))

            self.writer.add_scalars('data/seg_loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('data/predictor_loss', {
                'train': train_predictor_loss,
                'val': val_predictor_loss
            }, epoch)
            self.writer.add_scalars('data/dice', {
                'train': train_dice,
                'val': val_dice
            }, epoch)
            self.writer.add_scalars('data/run_dice', {
                'train': train_run_dice[0],
                'val': val_run_dice[0]
            }, epoch)

            self.writer.add_scalars('data/dice_dataratio', {
                'train': train_run_dice[0],
                'val': val_run_dice[0],
                'data_ratio': len(self.labeled_data_pool) / len(train_path),
            }, epoch)

            self.writer.add_scalar('data/lr', seg_optimizer.param_groups[0]['lr'], epoch)

            record_sample_list(
                record_csv=os.path.join(log_dir, 'sample_list.csv'),
                key=str(epoch),
                sample_list=self.labeled_data_pool)

            with open(os.path.join(log_dir, 'record.csv'), 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                rows = [epoch, len(self.labeled_data_pool) / len(train_path),
                        len(self.unlabeled_data_pool) / len(train_path), \
                        train_predictor_loss, val_predictor_loss, train_run_dice[0], val_run_dice[0]]
                rows.extend(train_run_dice[1][1:])
                rows.extend(val_run_dice[1][1:])
                csvwriter.writerow(rows)
            '''
            if val_loss < self.loss_threshold:
                self.loss_threshold = val_loss
            '''
            early_stopping(val_run_dice[0])

            # save
            if val_run_dice[0] > self.metrics_threshold:
                self.metrics_threshold = val_run_dice[0]

                if len(self.device.split(',')) > 1:
                    state_dict = net.module.state_dict()
                    predictor_state_dict = predictor.module.state_dict()
                else:
                    state_dict = net.state_dict()
                    predictor_state_dict = predictor.state_dict()

                saver = {
                    'epoch': epoch,
                    'save_dir': segnet_output_dir,
                    'state_dict': state_dict,
                    # 'optimizer':seg_optimizer.state_dict(), #TODO resume
                    # 'sample_count':sample_count
                }

                predictor_saver = {
                    'epoch': epoch,
                    'save_dir': predictor_output_dir,
                    'state_dict': predictor_state_dict,
                    # 'optimizer':predictor_optimizer.state_dict(), #TODO resume
                    # 'sample_count':sample_count
                }

                file_name = 'epoch={}-train_loss={:.5f}-train_dice={:.5f}-train_run_dice={:.5f}-val_loss={:.5f}-val_dice={:.5f}-val_run_dice={:.5f}.pth'.format(
                    epoch, train_loss, train_dice, train_run_dice[0], val_loss, val_dice, val_run_dice[0])
                predictor_file_name = 'epoch={}-train_predictor_loss={:.5f}-val_predictor_loss={:.5f}.pth'.format(
                    epoch, train_predictor_loss, val_predictor_loss)
                save_path = os.path.join(segnet_output_dir, file_name)
                predictor_save_path = os.path.join(predictor_output_dir, predictor_file_name)
                print("SegNet save as: %s" % file_name)
                print("Predictor save as: %s" % predictor_file_name)

                torch.save(saver, save_path)
                torch.save(predictor_saver, predictor_save_path)

            # early stopping
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.writer.close()
        dfs_remove_weight(segnet_output_dir, 3)
        dfs_remove_weight(predictor_output_dir, 3)
        print(f'total sample time is {total_sample_time}')
        print(f'sample time of every iter is {total_sample_time / self.sample_times}')

    def _get_data_loader(self, data_path=[], data_type='train', repeat_factor=1.0):

        assert len(data_path) != 0

        if data_type == 'train':
            data_transformer = transforms.Compose([
                Trunc_and_Normalize(self.scale, self.channels),
                Get_ROI(pad_flag=False)
                if self.get_roi else transforms.Lambda(lambda x: x),
                # CropResize(dim=self.input_shape,
                #         num_class=self.num_classes,
                #         crop=self.crop,
                #         channels=self.channels),
                RandomGenerator(self.input_shape),
                # RandomErase2D(scale_flag=False),
                # RandomZoom2D(), # bug for normalized MR image #TODO
                RandomDistort2D(),
                RandomRotate2D(),
                RandomFlip2D(mode='v'),
                # RandomAdjust2D(),
                RandomNoise2D(),
                To_Tensor1(num_class=self.num_classes, channels=self.channels)
            ])

        elif data_type == 'val':
            data_transformer = transforms.Compose([
                Trunc_and_Normalize(self.scale, self.channels),
                Get_ROI(pad_flag=False)
                if self.get_roi else transforms.Lambda(lambda x: x),
                # CropResize(dim=self.input_shape,
                #         num_class=self.num_classes,
                #         crop=self.crop,
                #         channels=self.channels),
                RandomGenerator(self.input_shape),
                To_Tensor0(num_class=self.num_classes, channels=self.channels)
            ])

        if data_type == 'train':
            dataset = DataGenerator(data_path,
                                    roi_number=self.roi_number,
                                    num_class=self.num_classes,
                                    transform=data_transformer,
                                    repeat_factor=repeat_factor)
        elif data_type == 'val':
            dataset = DataGeneratorval(data_path,
                                       roi_number=self.roi_number,
                                       num_class=self.num_classes,
                                       transform=data_transformer,
                                       repeat_factor=repeat_factor)

        # dataset = DataGenerator(data_path,
        #                         roi_number=self.roi_number,
        #                         num_class=self.num_classes,
        #                         transform=data_transformer,
        #                         repeat_factor=repeat_factor)

        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=data_type == 'train',
                                 num_workers=self.num_workers,
                                 pin_memory=True,
                                 drop_last=False)

        return data_loader

    def _random_sampling(self, sample_pool=[]):
        assert len(sample_pool) != 0
        samples = random.sample(sample_pool, k=int(len(sample_pool) * self.init_percent))
        return samples

    def _update_unlabeled_data_pool(self, labeled_data=[]):
        if len(labeled_data) != 0:
            for sample in labeled_data:
                self.unlabeled_data_pool.remove(sample)

    def _get_samples_per_epoch(self, N_sample, sample_times, sample_mode):

        init_sample = int(self.init_percent * N_sample)
        sample_strategy = np.zeros((sample_times + 1,))
        sample_strategy[0] = init_sample + 11

        n_sample_num = int(self.max_percent * N_sample - init_sample)
        first_sample = min(int(n_sample_num / (sample_times // 2)), init_sample)
        sample_strategy[1] = first_sample

        if sample_mode == "uniform":
            d = n_sample_num // sample_times
            for i in range(1, sample_times + 1):
                sample_strategy[i] = d

        elif sample_mode == "linear":
            d = int(2 * (sample_times * first_sample - n_sample_num) /
                    (sample_times * (sample_times - 1)))
            # print(d)
            for i in range(2, sample_times + 1):
                sample_strategy[i] = max(sample_strategy[i - 1] - d, 0)

        elif sample_mode == "convex":
            for sample_times in reversed(range(1, sample_times)):
                if 6 * (sample_times * first_sample - n_sample_num) / (
                        sample_times * (sample_times - 1) * (2 * sample_times - 1)) > (
                        sample_times - 1) ** 2 / first_sample:
                    break
            d = (9 * (sample_times * first_sample - n_sample_num) ** 2) / (4 * (sample_times - 1) ** 3)
            for t, i in enumerate(range(2, sample_times + 1)):
                sample_strategy[i] = int(first_sample - (d * t) ** 0.5)

        elif sample_mode == "square":
            for sample_times in reversed(range(1, sample_times)):
                if 6 * (sample_times * first_sample - n_sample_num) / (
                        sample_times * (sample_times - 1) * (2 * sample_times - 1)) > (
                        sample_times - 1) ** 2 / first_sample:
                    break
            d = 6 * (sample_times * first_sample - n_sample_num) / (
                        sample_times * (sample_times - 1) * (2 * sample_times - 1))
            for t, i in enumerate(range(2, sample_times + 1)):
                sample_strategy[i] = max(int(first_sample - d * t ** 2), 0)

        return sample_strategy

    def _train_on_epoch(self,
                        epoch,
                        net,
                        net2,
                        net3,
                        predictor,
                        criterion,
                        criteriond,
                        criteriondd,
                        predictor_criterion,
                        optimizer,
                        optimizer2,
                        predictor_optimizer,
                        scaler,
                        predictor_scaler,
                        train_loader=None,
                        sample_loader=None):

        net.train()
        net2.train()
        predictor.train()

        train_loss = AverageMeter()
        train_dice = AverageMeter()
        train_predictor_loss = AverageMeter()
        loader_train_iter = iter(sample_loader)

        from metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes), ignore_label=-1)
        iter_num = 0

        for step, sample in enumerate(train_loader):
            # try twice data_loader
            try:
                sample_u = next(loader_train_iter)

            except StopIteration:
                # loader_t_iter = iter(loader_train_t)
                # batch_t = next(loader_t_iter)
                loader_train_iter = iter(sample_loader)
                sample_u = next(loader_train_iter)

            # train seg net
            data = sample['image']  # N1HW
            target = sample['label']  # NCHW
            target_s = sample['label_s']
            # print('data.shape:', data.shape)
            # print('target.shape:', target.shape)

            ## pseudo create
            pseudo_labels, su_mask = label_propagation(data, target_s, 'ACDC')
            p_label = pseudo_labels.cuda()

            # load unlabeled data
            data_u = sample_u['image']
            data_u = data_u.cuda()

            data = data.cuda()
            target = target.cuda()
            target_s = target_s.cuda()

            data_h = torch.cat([data, data_u], dim=0)
            noise_h = torch.clamp(torch.randn_like(data_h) * 0.1, -0.2, 0.2)
            data_h_noise = data_h + noise_h


            with autocast(self.use_fp16):
                ##
                output = net(data_h)
                output2 = net2(data_h)
                output_noise = net(data_h_noise)
                output_soft = torch.softmax(output, dim=1)
                output2_soft = torch.softmax(output2, dim=1)
                output_noise_soft = torch.softmax(output_noise, dim=1)

                # print('data.shape:', data.shape)
                # print('data_h.shape:', data_h.shape)
                # print('output.shape:', output.shape)
                # print('output_soft.shape:', output_soft.shape)
                ##
                # output = net(data)
                # output2 = net2(data)
                # output_u = net(data_u)
                # output2_u = net2(data_u)
                # output_soft = torch.softmax(output, dim=1)
                # output2_soft = torch.softmax(output2, dim=1)
                # output_u_soft = torch.softmax(output_u, dim=1)
                # output2_u_soft = torch.softmax(output2_u, dim=1)
                #
                # output_noise = net(noise_input)
                # output_noise_soft = torch.softmax(output_noise, dim=1)

                with torch.no_grad():
                    ema_output = net3(data_h_noise)
                    ema_output_soft = torch.softmax(ema_output, dim=1)

                    # ema_output = net3(noise_input)
                    # ema_output_u = net2(noise_input_u)
                    # ema_output_soft = torch.softmax(ema_output, dim=1)
                    # ema_output_u_soft = torch.softmax(ema_output_u, dim=1)

                # cross perudo losses
                # pseudo_output = torch.argmax(output_soft.detach(), dim=1, keepdim=False)
                # pseudo_output2 = torch.argmax(output2_soft.detach(), dim=1, keepdim=False)

                pseudo_supervision1 = criteriond(output_soft[:data.shape[0]], output2_soft[:data.shape[0]])
                pseudo_supervision2 = criteriond(output2_soft[:data.shape[0]], output_soft[:data.shape[0]])  # .unsqueeze(1)
                pseudo_supervision3 = criteriond(output_soft[:data.shape[0]], output_noise_soft[:data.shape[0]])
                pseudo_supervision4 = criteriond(output_noise_soft[:data.shape[0]], output_soft[:data.shape[0]])
                pseudo_supervision5 = criteriond(output2_soft[:data.shape[0]], output_noise_soft[:data.shape[0]])
                pseudo_supervision6 = criteriond(output_noise_soft[:data.shape[0]], output2_soft[:data.shape[0]])
                pseudo_supervision_noise = pseudo_supervision3 + pseudo_supervision4 + pseudo_supervision5 + pseudo_supervision6

                pseudo_supervision1_u = criteriond(output_soft[data.shape[0]:], output2_soft[data.shape[0]:])
                pseudo_supervision2_u = criteriond(output2_soft[data.shape[0]:], output_soft[data.shape[0]:])  # .unsqueeze(1)
                pseudo_supervision_u = pseudo_supervision1_u + pseudo_supervision2_u

                # chaoxiangsu pseudo
                loss_p1 = criteriondd(output_soft[:data.shape[0]], p_label)
                loss_p2 = criteriondd(output2_soft[:data.shape[0]], p_label)
                loss_p_noise = criteriondd(output_noise_soft[:data.shape[0]], p_label)
                loss_p = loss_p1 + loss_p2 + loss_p_noise

                # with ema_model online
                loss_ema_l = criteriond(output_soft[:data.shape[0]], ema_output_soft[:data.shape[0]]) + criteriond(output2_soft[:data.shape[0]], ema_output_soft[:data.shape[0]])  # + criteriond(output_noise_soft, ema_output_soft)
                loss_ema_u = criteriond(output_soft[data.shape[0]:], ema_output_soft[data.shape[0]:]) + criteriond(output2_soft[data.shape[0]:], ema_output_soft[data.shape[0]:])
                loss_ema = loss_ema_l  # + loss_ema_l

                if epoch >= (self.warmup_epoch - self.sample_inteval):
                    cp1 = 0.0  # before 0.0
                else:
                    cp1 = 0.5

                if epoch <= 60:
                    cp2 = 0.0
                else:
                    cp2 = 0.5

                loss = 0.6 * criterion(output[:data.shape[0]], target_s.long()) + 0.4 * criterion(output2[:data.shape[0]],target_s.long()) + 0.2 * criterion(
                    output_noise[:data.shape[0]], target_s.long()) + cp1 * (pseudo_supervision1 + pseudo_supervision2) + 0.3 * loss_p + 1.5 * cp1 * loss_ema + cp2 * (loss_ema_u + pseudo_supervision_u)

                if isinstance(output, tuple):
                    output = output[0]  # NCHW

            optimizer.zero_grad()
            optimizer2.zero_grad()
            if self.use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.step(optimizer2)
                scaler.update()

            else:
                loss.backward()
                optimizer.step()
                optimizer2.step()

            update_ema_variables(net2, net3, 0.99, iter_num)
            iter_num = iter_num + 1

            ## train predictor
            if epoch >= (self.warmup_epoch - self.sample_inteval):

                predictor_target = torch.from_numpy(
                    compute_dice(output[:data.shape[0]].detach(), target, ignore_index=-1, reduction=None)).cuda()  # NC
                predictor_data = torch.stack([torch.cat((i, j), dim=0) for i, j in zip(data.clone().detach(),
                                                                                       torch.softmax(
                                                                                           output[:data.shape[0]].clone().detach(),
                                                                                           dim=1))], dim=0)  # N,C+1,H,W

                with autocast(self.use_fp16):
                    predictor_output = predictor(predictor_data)
                    predictor_loss = predictor_criterion(predictor_output, predictor_target)

                predictor_output = predictor_output.mean(dim=0).detach().cpu().numpy().tolist()

                predictor_optimizer.zero_grad()
                if self.use_fp16:
                    predictor_scaler.scale(predictor_loss).backward()
                    predictor_scaler.step(predictor_optimizer)
                    predictor_scaler.update()

                else:
                    predictor_loss.backward()
                    predictor_optimizer.step()
            else:
                predictor_target = torch.from_numpy(
                    np.ones((data.size(0), self.num_classes), dtype=np.float32) * -1.0)  # NC
                predictor_output = [-1.0] * self.num_classes
                predictor_loss = torch.tensor(-1.0).cuda()

            output = output[:data.shape[0]].float()
            loss = loss.float()
            predictor_loss = predictor_loss.float()

            # measure dice and record loss
            dice = compute_dice(output[:data.shape[0]].detach(), target)
            train_loss.update(loss.item(), data.size(0))
            train_dice.update(dice, data.size(0))
            train_predictor_loss.update(predictor_loss.item(), data.size(0))

            # measure run dice
            output = torch.argmax(torch.softmax(output[:data.shape[0]], dim=1), 1).detach().cpu().numpy()  # N*H*W
            target = torch.argmax(target, 1).detach().cpu().numpy()
            run_dice.update_matrix(target, output)

            torch.cuda.empty_cache()

            if self.global_step % 2 == 0:
                rundice, dice_list = run_dice.compute_dice()
                print("Category Dice: ", np.round(predictor_target.cpu().numpy().mean(axis=0), 4))
                print("Predicted Dice: ", np.round(np.array(predictor_output), 4))
                print(
                    'epoch:{},step:{},train_loss:{:.5f},train_dice:{:.5f},train_run_dice:{:.5f},train_predictor_loss:{:.5f},lr:{:.5f}'
                    .format(epoch, step, loss.item(), dice, rundice,
                            predictor_loss.item(),
                            optimizer.param_groups[0]['lr']))
                # run_dice.init_op()
                self.writer.add_scalars(
                    'data/train_loss_dice', {
                        'train_loss': loss.item(),
                        'train_dice': dice,
                        'train_predictor_loss': predictor_loss.item()
                    }, self.global_step)

            self.global_step += 1

        return train_loss.avg, train_dice.avg, run_dice.compute_dice(), train_predictor_loss.avg

    def _val_on_epoch(self,
                      epoch,
                      net,
                      predictor,
                      criterion,
                      predictor_criterion,
                      val_loader=None):

        net.eval()
        predictor.eval()

        val_loss = AverageMeter()
        val_dice = AverageMeter()
        val_predictor_loss = AverageMeter()

        from metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes), ignore_label=-1)

        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()
                with autocast(self.use_fp16):
                    output = net(data)
                    loss = criterion(output, target)

                    if isinstance(output, tuple):
                        output = output[0]

                if epoch >= (self.warmup_epoch - self.sample_inteval):
                    predictor_target = torch.from_numpy(
                        compute_dice(output.detach(), target, ignore_index=-1, reduction=None)).cuda()  # NC
                    predictor_data = torch.stack([torch.cat((i, j), dim=0) for i, j in zip(data.clone().detach(),
                                                                                           torch.softmax(
                                                                                               output.clone().detach(),
                                                                                               dim=1))],
                                                 dim=0)  # N,C+1,H,W

                    with autocast(self.use_fp16):
                        predictor_output = predictor(predictor_data)
                        predictor_loss = predictor_criterion(predictor_output, predictor_target)

                    predictor_output = predictor_output.mean(dim=0).detach().cpu().numpy().tolist()
                else:
                    # predictor_target = torch.from_numpy(np.ones(data.size(0),self.num_classes)*-1.0)
                    predictor_target = torch.from_numpy(
                        np.ones((data.size(0), self.num_classes), dtype=np.float32) * -1.0)  # NC
                    predictor_output = [-1.0] * self.num_classes
                    predictor_loss = torch.tensor(-1.0).cuda()

                output = output.float()
                loss = loss.float()
                predictor_loss = predictor_loss.float()

                # measure dice and record loss
                dice = compute_dice(output.detach(), target)
                val_loss.update(loss.item(), data.size(0))
                val_dice.update(dice, data.size(0))
                val_predictor_loss.update(predictor_loss.item(), data.size(0))

                # measure run dice
                output = torch.argmax(torch.softmax(output, dim=1), 1).detach().cpu().numpy()  # N*H*W
                target = torch.argmax(target, 1).detach().cpu().numpy()
                run_dice.update_matrix(target, output)

                torch.cuda.empty_cache()

                if step % 2 == 0:
                    rundice, dice_list = run_dice.compute_dice()
                    print("Category Dice: ", np.round(predictor_target.cpu().numpy().mean(axis=0), 4))
                    print("Predicted Dice: ", np.round(np.array(predictor_output), 4))
                    print(
                        'epoch:{},step:{},val_loss:{:.5f},val_dice:{:.5f},val_run_dice:{:.5f},val_predictor_loss:{:.5f}'
                        .format(epoch, step, loss.item(), dice, rundice, predictor_loss.item()))
                    # run_dice.init_op()

        # return val_loss.avg,run_dice.compute_dice()[0]
        return val_loss.avg, val_dice.avg, run_dice.compute_dice(), val_predictor_loss.avg

    def _get_seg_net(self, net_name):

        if net_name == 'unet':
            if self.encoder_name is None:
                raise ValueError("encoder name must not be 'None'!")
            else:
                if self.encoder_name in ['resnet50_dropout']:
                    from model.unet import unet
                    net = unet(net_name,
                               encoder_name=self.encoder_name,
                               in_channels=self.channels,
                               classes=self.num_classes
                               )
                else:
                    import segmentation_models_pytorch as smp
                    net = smp.Unet(encoder_name=self.encoder_name,
                                   encoder_weights=None
                                   if not self.ex_pre_trained else 'imagenet',
                                   in_channels=self.channels,
                                   classes=self.num_classes)
        elif net_name == 'unet++':
            if self.encoder_name is None:
                raise ValueError("encoder name must not be 'None'!")
            else:
                import segmentation_models_pytorch as smp
                net = smp.UnetPlusPlus(encoder_name=self.encoder_name,
                                       encoder_weights=None if
                                       not self.ex_pre_trained else 'imagenet',
                                       in_channels=self.channels,
                                       classes=self.num_classes)

        elif net_name == 'vit':
            net = ViT_seg(config, img_size=224, num_classes=4).cuda()
            net.load_from(config)

        elif net_name == 'vit_m':
            net = ViT_seg(config, img_size=224, num_classes=4).cuda()
            net.load_from(config)
            for param in net.parameters():
                param.detach()

        elif net_name == 'FPN':
            if self.encoder_name is None:
                raise ValueError("encoder name must not be 'None'!")
            else:
                import segmentation_models_pytorch as smp
                net = smp.FPN(encoder_name=self.encoder_name,
                              encoder_weights=None
                              if not self.ex_pre_trained else 'imagenet',
                              in_channels=self.channels,
                              classes=self.num_classes)

        elif net_name == 'deeplabv3+':
            if self.encoder_name is None:
                raise ValueError("encoder name must not be 'None'!")
            else:
                import segmentation_models_pytorch as smp
                net = smp.DeepLabV3Plus(
                    encoder_name=self.encoder_name,
                    encoder_weights=None
                    if not self.ex_pre_trained else 'imagenet',
                    in_channels=self.channels,
                    classes=self.num_classes)

        return net

    def _get_predictor(self, predictor_name):
        if predictor_name.startswith('ap'):
            import model.predictor as predictor
            predictor = predictor.__dict__[predictor_name](
                input_channels=self.channels + self.num_classes,
                num_classes=self.num_classes,
                final_drop=0.5)

        return predictor

    def _get_seg_loss(self, loss_fun, class_weight=None):
        if class_weight is not None:
            class_weight = torch.tensor(class_weight)

        if loss_fun == 'Cross_Entropy':
            from loss.cross_entropy import CrossentropyLoss
            loss = CrossentropyLoss(weight=class_weight)

        elif loss_fun == 'TopKLoss':
            from loss.cross_entropy import TopKLoss
            loss = TopKLoss(weight=class_weight, k=self.topk)

        elif loss_fun == 'CELabelSmoothingPlusDice':
            from loss.combine_loss import CELabelSmoothingPlusDice
            loss = CELabelSmoothingPlusDice(smoothing=0.1,
                                            weight=class_weight,
                                            ignore_index=0)

        elif loss_fun == 'OHEM':
            from loss.cross_entropy import OhemCELoss
            loss = OhemCELoss(thresh=0.7)

        elif loss_fun == 'DiceLoss':
            from loss.dice_loss import DiceLoss
            loss = DiceLoss(weight=class_weight, ignore_index=0, p=1)

        elif loss_fun == 'CEPlusDice':
            from loss.combine_loss import CEPlusDice
            loss = CEPlusDice(weight=class_weight, ignore_index=0)

        return loss

    def _get_predictor_loss(self, predictor_loss_fun):

        if predictor_loss_fun == 'MSE':
            loss = torch.nn.MSELoss()

        return loss

    def _get_optimizer(self, optimizer, net, lr):
        """
        Build optimizer, set weight decay of normalization to 0 by default.
        """

        def check_keywords_in_name(name, keywords=()):
            isin = False
            for keyword in keywords:
                if keyword in name:
                    isin = True
            return isin

        def set_weight_decay(model, skip_list=(), skip_keywords=()):
            has_decay = []
            no_decay = []

            for name, param in model.named_parameters():
                # check what will happen if we do not set no_weight_decay
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                        check_keywords_in_name(name, skip_keywords):
                    no_decay.append(param)
                    # print(f"{name} has no weight decay")
                else:
                    has_decay.append(param)
            return [{
                'params': has_decay
            }, {
                'params': no_decay,
                'weight_decay': 0.
            }]

        skip = {}
        skip_keywords = {}
        if hasattr(net, 'no_weight_decay'):
            skip = net.no_weight_decay()
        if hasattr(net, 'no_weight_decay_keywords'):
            skip_keywords = net.no_weight_decay_keywords()
        parameters = set_weight_decay(net, skip, skip_keywords)

        opt_lower = optimizer.lower()
        optimizer = None
        if opt_lower == 'sgd':
            optimizer = torch.optim.SGD(parameters,
                                        momentum=self.momentum,
                                        nesterov=True,
                                        lr=lr,
                                        weight_decay=self.weight_decay)
        elif opt_lower == 'adamw':
            optimizer = torch.optim.AdamW(parameters,
                                          eps=1e-8,
                                          betas=(0.9, 0.999),
                                          lr=lr,
                                          weight_decay=self.weight_decay)
        elif opt_lower == 'adam':
            optimizer = torch.optim.Adam(parameters,
                                         lr=lr,
                                         weight_decay=self.weight_decay)

        return optimizer

    def _get_lr_scheduler(self, lr_scheduler, optimizer):
        if lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, verbose=True)
            return lr_scheduler

        elif lr_scheduler == 'CustomScheduler':
            from custom_scheduler import CustomScheduler
            lr_scheduler = CustomScheduler(
                optimizer=optimizer,
                max_lr=self.lr,
                min_lr=1e-6,
                lr_warmup_steps=self.warmup_epoch,
                lr_decay_steps=self.n_epoch,
                lr_decay_style='cosine',
                start_wd=self.weight_decay,
                end_wd=self.weight_decay,
                wd_incr_style='constant',
                wd_incr_steps=self.n_epoch
            )
            return lr_scheduler

        elif lr_scheduler == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.milestones, gamma=self.gamma)
            return lr_scheduler

        elif lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.n_epoch, eta_min=1e-6)
            return lr_scheduler

        elif lr_scheduler == 'CosineAnnealingWarmRestarts':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 20, T_mult=2)
            return lr_scheduler

    def _get_pre_trained_seg_net(self, weight_path, ckpt_point=True):
        checkpoint = torch.load(weight_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        if ckpt_point:
            self.start_epoch = checkpoint['epoch'] + 1
            self.metrics_threshold = eval(
                weight_path.split('=')[-1].split('.')[0])

    def _get_pre_trained_predictor(self, weight_path, ckpt_point=True):
        checkpoint = torch.load(weight_path)
        self.predictor.load_state_dict(checkpoint['state_dict'])


# computing tools


class AverageMeter(object):
    '''
  Computes and stores the average and current value
  '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def binary_dice(predict, target, smooth=1e-5, reduction='mean'):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1e-5
        predict: A numpy array of shape [N, *]
        target: A numpy array of shape same with predict
    Returns:
        DSC numpy array according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    assert predict.shape[0] == target.shape[
        0], "predict & target batch size don't match"
    predict = predict.reshape(predict.shape[0], -1)  # N，H*W
    target = target.reshape(target.shape[0], -1)  # N，H*W

    inter = np.sum(np.multiply(predict, target), axis=1)  # N
    union = np.sum(predict + target, axis=1)  # N

    dice = (2 * inter + smooth) / (union + smooth)  # N

    if reduction == 'mean':
        # nan mean
        dice_index = dice != 1.0
        dice = dice[dice_index]
        return dice.mean()
    else:
        return dice  # N


def compute_dice(predict, target, ignore_index=0, reduction='mean'):
    """
    Compute dice
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        ignore_index: class index to ignore
    Return:
        mean dice over the batch
    """
    N, num_classes, _, _ = predict.size()
    assert predict.shape == target.shape, 'predict & target shape do not match'
    predict = F.softmax(predict, dim=1)

    predict = torch.argmax(predict, dim=1).detach().cpu().numpy()  # N*H*W
    target = torch.argmax(target, dim=1).detach().cpu().numpy()  # N*H*W

    if reduction == 'mean':
        dice_array = -1.0 * np.ones((num_classes,), dtype=np.float32)  # C
    else:
        dice_array = -1.0 * np.ones((num_classes, N), dtype=np.float32)  # CN

    for i in range(num_classes):
        if i != ignore_index:
            if i not in predict and i not in target:
                continue
            dice = binary_dice((predict == i).astype(np.float32),
                               (target == i).astype(np.float32),
                               reduction=reduction)
            dice_array[i] = np.round(dice, 4)

    if reduction == 'mean':
        dice_array = np.where(dice_array == -1.0, np.nan, dice_array)
        return np.nanmean(dice_array[1:])
    else:
        dice_array = np.where(dice_array == -1.0, 1.0,
                              dice_array).transpose(1, 0)  # CN -> NC
        return dice_array


class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""

    def __init__(self,
                 patience=10,
                 verbose=True,
                 delta=0,
                 monitor='val_loss',
                 op_type='min'):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type

        if self.op_type == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):

        score = -val_score if self.op_type == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0

    def print_and_update(self, val_score):
        '''print_message when validation score decrease.'''
        if self.verbose:
            print(
                self.monitor,
                f'optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...'
            )
        self.val_score_min = val_score