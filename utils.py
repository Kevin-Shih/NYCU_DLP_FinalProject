import numpy as np
import torchvision.utils as vutils
import torch, random
import torch.nn.functional as F
import cv2
from typing import List, Union, Tuple, Dict
# from refile import *
import io
import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import collections
import re

class NanError(Exception):
    pass

def recursive_apply(obj: Union[List, Dict], func):
    assert type(obj) == dict or type(obj) == list
    idx_iter = obj if type(obj) == dict else range(len(obj))
    for k in idx_iter:
        if type(obj[k]) == list or type(obj[k]) == dict:
            recursive_apply(obj[k], func)
        else:
            obj[k] = func(obj[k])

def visualize_depth(depth, mask=None, depth_min=None, depth_max=None, direct=False):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    if not direct:
        depth = 1.0 / (depth + 1e-6)
    invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
    if mask is not None:
        invalid_mask += np.logical_not(mask)
    if depth_min is None:
        depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
    if depth_max is None:
        depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
    depth_color[invalid_mask, :] = 0

    return depth_color


def save_model_vis(obj, save_dir: str, job_name: str, global_step: int, max_keep: int):
    os.makedirs(os.path.join(save_dir, job_name), exist_ok=True)
    record_file = os.path.join(save_dir, job_name, 'record')
    cktp_file = os.path.join(save_dir, job_name, f'{global_step}.tar')
    if not os.path.exists(record_file):
        with open(record_file, 'w+') as f:
            json.dump([], f)
    with open(record_file, 'r') as f:
        record = json.load(f)
    record.append(global_step)
    if len(record) > max_keep:
        old = record[0]
        record = record[1:]
        os.remove(os.path.join(save_dir, job_name, f'{old}.tar'))
    torch.save(obj, cktp_file)
    with open(record_file, 'w') as f:
        json.dump(record, f)


def load_model_vis(model: nn.Module, load_path: str, load_step: int):
    model.load_state_dict(torch.load(load_path)['model'])
    return 0


# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return torch.mean(err_mask.float())


# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def AbsDepthError_metrics(depth_est, depth_gt, mask, thres=None):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    error = (depth_est - depth_gt).abs()
    if thres is not None:
        error = error[(error >= float(thres[0])) & (error <= float(thres[1]))]
        if error.shape[0] == 0:
            return torch.tensor(0, device=error.device, dtype=error.dtype)
    return torch.mean(error)

import torch.distributed as dist
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def reduce_scalar_outputs(scalar_outputs):
    world_size = get_world_size()
    if world_size < 2:
        return scalar_outputs
    with torch.no_grad():
        names = []
        scalars = []
        for k in sorted(scalar_outputs.keys()):
            names.append(k)
            scalars.append(scalar_outputs[k])
        scalars = torch.stack(scalars, dim=0)
        dist.reduce(scalars, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            scalars /= world_size
        reduced_scalars = {k: v for k, v in zip(names, scalars)}

    return reduced_scalars

import torch
from bisect import bisect_right
# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        #print("base_lr {}, warmup_factor {}, self.gamma {}, self.milesotnes {}, self.last_epoch{}".format(
        #    self.base_lrs[0], warmup_factor, self.gamma, self.milestones, self.last_epoch))
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def local_pcd(depth, intr):
    nx = depth.shape[1]  # w
    ny = depth.shape[0]  # h
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    x = x.reshape(nx * ny)
    y = y.reshape(nx * ny)
    p2d = np.array([x, y, np.ones_like(y)])
    p3d = np.matmul(np.linalg.inv(intr), p2d)
    depth = depth.reshape(1, nx * ny)
    p3d *= depth
    p3d = np.transpose(p3d, (1, 0))
    p3d = p3d.reshape(ny, nx, 3).astype(np.float32)
    return p3d

def generate_pointcloud(rgb, depth, ply_file, intr, scale=1.0):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file

    """
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    points = []
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v, u] #rgb.getpixel((u, v))
            Z = depth[v, u] / scale
            if Z == 0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
    file = open(ply_file, "w")
    file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            ''' % (len(points), "".join(points)))
    file.close()
    print("save ply, fx:{}, fy:{}, cx:{}, cy:{}".format(fx, fy, cx, cy))

        
def flatten_with_shape(tensor_list, mask_list):
    """
    Params:
    :tensor_list: [(B, C, H1, W1), ..., (B, C, HN, WN)]
    :mask_list: [(B, H1, W1), ..., (B, HN, WN)]

    Return:
    :tensor_flatten: (B, L, C)
    :mask_flatten: (B, L)
    :tensor_shape: (N, 2)
    """
    assert isinstance(tensor_list, collections.abc.Sequence)
    assert len(tensor_list) > 0

    N = len(tensor_list)
    tensor_shape = torch.zeros(N, 2, dtype=torch.int64, device=tensor_list[0].device)
    tensor_flatten = []

    if mask_list is not None:
        mask_flatten = []

    for i, tensor in enumerate(tensor_list):
        new_tensor = tensor.flatten(2).permute(0, 2, 1)
        tensor_flatten.append(new_tensor)

        if mask_list is not None:
            mask = mask_list[i]
            new_mask = mask.flatten(1)
            mask_flatten.append(new_mask)
            assert tensor.shape[2] == mask.shape[1]
            assert tensor.shape[3] == mask.shape[2]
        tensor_shape[i, 0] = tensor.shape[2]
        tensor_shape[i, 1] = tensor.shape[3]

    mask_flatten = torch.cat(mask_flatten, dim=1) if mask_list is not None else None
    tensor_flatten = torch.cat(tensor_flatten, dim=1)

    return tensor_flatten, mask_flatten, tensor_shape


def view_with_shape(tensor_flatten, mask_flatten, tensor_shape):
    """
    Params:
    :tensor_flatten: (B, L, C)
    :mask_flatten: (B, L)
    :tensor_shape: (N, 2)

    Return:
    :tensor_list: [(B, C, H1, W1), ..., (B, C, HN, WN)]
    :mask_list: [(B, H1, W1), ..., (B, HN, WN)]
    """
    chunk_sizes = (tensor_shape[:, 0] * tensor_shape[:, 1]).tolist()
    N = tensor_shape.shape[0]

    if tensor_flatten is None and mask_flatten is None:
        raise ValueError("Both tensor and mask are None")
    B = tensor_flatten.shape[0] if tensor_flatten is not None else mask_flatten.shape[0]

    if tensor_flatten is not None:
        tensor_list = torch.split(tensor_flatten, chunk_sizes, dim=1)

    if mask_flatten is not None:
        mask_list = torch.split(mask_flatten, chunk_sizes, dim=1)

    tensor2d_list = [] if tensor_flatten is not None else None
    mask2d_list = [] if mask_flatten is not None else None
    for i in range(N):
        H, W = tensor_shape[i].tolist()
        if tensor_flatten is not None:
            tensor2d_list.append(
                tensor_list[i].view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            )
        if mask_flatten is not None:
            mask2d_list.append(mask_list[i].view(B, H, W))

    return tensor2d_list, mask2d_list


def split_with_shape(tensor_flatten, mask_flatten, tensor_shape):
    """
    Params:
    :tensor_flatten: (B, L, C)
    :mask_flatten: (B, L)
    :tensor_shape: (N, 2)

    Return:
    :tensor_list: [(B, H1 * W1, C), ..., (B, HN * WN, C)]
    :mask_list: [(B, H1 * W1), ..., (B, HN * WN)]
    """
    chunk_sizes = (tensor_shape[:, 0] * tensor_shape[:, 1]).tolist()

    if tensor_flatten is None and mask_flatten is None:
        raise ValueError("Both tensor and mask are None")

    if tensor_flatten is not None:
        tensor_list = torch.split(tensor_flatten, chunk_sizes, dim=1)
    else:
        tensor_list = None

    if mask_flatten is not None:
        mask_list = torch.split(mask_flatten, chunk_sizes, dim=1)
    else:
        mask_list = None

    return tensor_list, mask_list


# np_str_obj_array_pattern = re.compile(r"[SaUO]")