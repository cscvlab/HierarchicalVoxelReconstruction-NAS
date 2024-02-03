import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree as KDTree
import os
import sys
sys.path.append('./scripts')

def compute_chamfer(predict_coor,ground_truth_coor):
    if predict_coor.size == 0:
        return 1e5
    if predict_coor.size > ground_truth_coor.size * 10:
        return 1e5
    # one direction
    gen_points_kd_tree = KDTree(predict_coor)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(ground_truth_coor)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(ground_truth_coor)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(predict_coor)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer

def compute_iou(pred: torch.Tensor, gt: torch.Tensor):
    pred, gt = pred > 0.5, gt > 0.5
    intersection = torch.sum(pred & gt)
    union = torch.sum(pred | gt)
    return intersection / (union + 1e-10)

def compute_accuracy(pred, gt):
    pred, gt = pred > 0.5, gt > 0.5
    t = torch.sum(pred == gt)
    s = pred.shape[0]
    return t / s

def compute_precision(pred, gt):
    pred, gt = pred > 0.5, gt > 0.5
    tp = torch.sum(pred & gt)
    fp = torch.sum(~pred & gt)
    return tp / (tp + fp + 1e-10)

def compute_recall(pred, gt):
    pred, gt = pred > 0.5, gt > 0.5
    tp = torch.sum(pred & gt)
    fn = torch.sum(pred & ~gt)
    return tp / (tp + fn + 1e-10)

def compute_pts_iou(pred, gt):
    """
    pred: predict point cloud. Shape: [N_points, 3]
    gt: ground truth point cloud. Shape: [M_points, 3]
    """
    if pred.shape[0] == 0:
        return torch.Tensor([0])
    box_pred_max = torch.max(pred, dim=0)[0]
    box_pred_min = torch.min(pred, dim=0)[0]
    box_gt_max = torch.max(gt, dim=0)[0]
    box_gt_min = torch.min(gt, dim=0)[0]
    
    volume1 = box_pred_max - box_pred_min
    volume1 = volume1[0] * volume1[1] * volume1[2]
    volume2 = box_gt_max - box_gt_min
    volume2 = volume2[0] * volume2[1] * volume2[2]
    
    interX = min((box_pred_max[0] - box_gt_min[0]).item(), (box_gt_max[0] - box_pred_min[0]).item())
    interY = min((box_pred_max[1] - box_gt_min[1]).item(), (box_gt_max[1] - box_pred_min[1]).item())
    interZ = min((box_pred_max[2] - box_gt_min[2]).item(), (box_gt_max[2] - box_pred_min[2]).item())
    intersection = interX * interY * interZ
    
    iou = intersection / (volume1 + volume2 - intersection + 1e-10)
    return iou

def compute_binary_metrics(pred: torch.Tensor, gt: torch.Tensor, threshold = 0.5):
    """
    Args:
        pred: pred classification 
        gt: ground truth 
    """
    pred, gt = pred > threshold, gt > threshold
    pred = pred.to(gt.device)
    s = 1
    for r in pred.shape:
        s *= r
    t = torch.sum(pred == gt)
    tp = torch.sum(pred & gt)
    fp = torch.sum(~pred & gt)
    fn = torch.sum(pred & ~gt)
    union = torch.sum(pred | gt)
    
    accuracy = t / s
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    iou = tp / (union + 1e-10)
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'iou': iou.item(),
        'num': s,
        'tp': tp.item(),
        'fp': fp.item(),
        'fn': fn.item()
    }

    
