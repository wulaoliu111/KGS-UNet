import argparse
import os
def _align_input_channels(input_tensor, net):

    if input_tensor.dim() != 4:
        return input_tensor
    expected_c = None
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            expected_c = m.in_channels
            break
    if expected_c is None:
        return input_tensor
    cur_c = input_tensor.shape[1]
    if cur_c == expected_c:
        return input_tensor
    if cur_c == 1 and expected_c == 3:
        return input_tensor.repeat(1, 3, 1, 1)
    if cur_c == 3 and expected_c == 1:
        return input_tensor[:, :1, ...]
    raise RuntimeError(f"Cannot automatically align channels: input C={cur_c}, expected C={expected_c}")

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""
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





import torch

import torch.nn as nn
import numpy as np
from medpy import metric
from scipy.ndimage import zoom
import seaborn as sns

import SimpleITK as sitk
import pandas as pd


from thop import profile
from thop import clever_format


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP]([URL]
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
    
def one_hot_encoder(input_tensor,dataset,n_classes = None):
    tensor_list = []
    if dataset == 'MMWHS':  
        dict = [0,205,420,500,550,600,820,850]
        for i in dict:
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    else:
        for i in range(n_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()    

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
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
        #print(inputs)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, hd95, jaccard, asd
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0

def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    else:
        return 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1,do_deeps=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    device = next(net.parameters()).device if next(net.parameters(), None) is not None else torch.device('cpu')
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
            input = _align_input_channels(input, net)
            #input = input.repeat(1, 3, 1, 1)
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                if do_deeps:
                    outputs = outputs[-1]

                if outputs.shape[1] == 1:
                    prob = torch.sigmoid(outputs)
                    out = (prob > 0.5).long()
                else:
                    out = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        x, y = image.shape[0], image.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().to(device)
        input = _align_input_channels(input, net)
        #input = input.repeat(1, 3, 1, 1)
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            if x != patch_size[0] or y != patch_size[1]:
                out = out.cpu().detach().numpy()
                prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    #if test_save_path is not None:
    #    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    #    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    #    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    #    img_itk.SetSpacing((1, 1, z_spacing))
    #    prd_itk.SetSpacing((1, 1, z_spacing))
    #    lab_itk.SetSpacing((1, 1, z_spacing))
    #    sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
    #    sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
    #    sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def val_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    device = next(net.parameters()).device if next(net.parameters(), None) is not None else torch.device('cpu')
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
            input = _align_input_channels(input, net)
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                #out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                if outputs.shape[1] == 1:
                    prob = torch.sigmoid(outputs)
                    out = (prob > 0.5).long()
                else:
                    out = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
                x, y = image.shape[0], image.shape[1]
                if x != patch_size[0] or y != patch_size[1]:
                    image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
                input = torch.from_numpy(image).unsqueeze(
                    0).unsqueeze(0).float().to(device)
                input = _align_input_channels(input, net)
                # input = input.repeat(1, 3, 1, 1)
                net.eval()
                with torch.no_grad():
                    outputs = net(input)
                    out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                    if x != patch_size[0] or y != patch_size[1]:
                        out = out.cpu().detach().numpy()
                        prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                    else:
                        prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    return metric_list


import torch
import torch.nn as nn

def align_pred_target_for_loss(pred: torch.Tensor,
                               target: torch.Tensor,
                               criterion: nn.Module | None = None):


    if isinstance(pred, (list, tuple)):
        pred = pred[-1]


    if isinstance(criterion, (nn.BCEWithLogitsLoss, nn.BCELoss)):
        # pred: [B,1,H,W]
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        elif pred.dim() == 2:
            pred = pred.view(pred.size(0), 1, *target.shape[-2:])
        elif pred.dim() == 4 and pred.size(1) != 1:

            pred = pred[:, :1, ...]
        # target: [B,1,H,W] 且 float (0/1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if target.dtype != torch.float32 and target.dtype != torch.float16 and target.dtype != torch.bfloat16:
            target = target.float()

        target = (target > 0).to(pred.dtype)
        if pred.shape != target.shape:

            if target.shape[-2:] != pred.shape[-2:]:
                target = torch.nn.functional.interpolate(
                    target, size=pred.shape[-2:], mode='nearest')

            if target.size(0) != pred.size(0):
                M = min(target.size(0), pred.size(0))
                pred, target = pred[:M], target[:M]
        return pred, target


    if isinstance(criterion, nn.CrossEntropyLoss):

        if pred.dim() == 3:

            pred = pred.unsqueeze(1)
        if pred.dim() != 4 or pred.size(1) < 2:
            raise ValueError(
                "CrossEntropyLoss 需要 pred=[B,C,H,W] 且 C>=2；"
                "当前模型是二分类 1 通道，请将 num_classes 改为 2 或改用 BCEWithLogitsLoss。"
            )

        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        if target.dim() != 3:
            raise ValueError("CrossEntropyLoss 需要 target=[B,H,W]（整型类别索引）")
        if target.dtype != torch.long:
            target = target.long()

        if pred.shape[-2:] != target.shape[-2:]:
            target = torch.nn.functional.interpolate(
                target.unsqueeze(1).float(), size=pred.shape[-2:], mode='nearest').squeeze(1).long()
        return pred, target


    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    return pred, target
