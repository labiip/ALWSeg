import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
# from networks.net_factory import net_factory

# from networks.vision_transformer import SwinUnet as ViT_seg
# from networks.net_factory import net_factory, config, args
from model.unet import unet
import segmentation_models_pytorch as smp
from model.utils import count_params_and_macs
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 0"

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/MSCMR', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='MSCMR_al/unet', help='experiment_name')  # Uncertainty_Rectified_Pyramid_Consistency_7_labeled /Cross_Teaching_Between_CNN_Transformer_7/Cross_Pseudo_Supervision
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=14,
                    help='labeled data')


# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     dice = metric.binary.dc(pred, gt)
#     asd = metric.binary.asd(pred, gt)
#     # asd = 0
#     hd95 = metric.binary.hd95(pred, gt)
#     # hd95 = 0
#     return dice, hd95, asd

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95, asd
    else:
        return 0, 50, 10


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice/1.0).unsqueeze(
            0).unsqueeze(0).float()  #.cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _= net(input)
            else:
                # out_main, _, _, _, _, _, _ = net(input)
                out_main, _ = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)

            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)     # while num_classes = 1, every prediction == 1 && label ==1
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "./model/{}/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "./model/{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    # net = unet('unet', encoder_name='resnet50_naive', in_channels=1, classes=4, aux_losspredictor=False)
    # net = smp.Unet('unet', encoder_name='resnet50_naive', in_channels=1, classes=4, aux_losspredictor=False)
    net = smp.Unet(encoder_name='resnet50',
                   encoder_weights=None,
                   in_channels=1,
                   classes=4)
    # net = net_factory(net_type=FLAGS.model, in_chns=1,
    #                   class_num=FLAGS.num_classes)

    # net = ViT_seg(config, img_size=[224, 224], num_classes=args.num_classes).cuda()
    # net.load_from(config)

    save_mode_path = os.path.join(
        snapshot_path, '186.pth'.format(FLAGS.model)) # '{}_best_model1.pth'

    net.load_state_dict(torch.load(save_mode_path)['state_dict'])

    # state_dict = torch.load(save_mode_path)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
