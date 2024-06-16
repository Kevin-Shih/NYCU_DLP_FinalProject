# from refile import *
import torch
import collections
        
def flatten_with_shape(tensor_list, mask_list):
    """
    Params:
    :tensor_list: [(B, C, H1, W1), ..., (B, C, HN, WN)]
    :mask_list: [(B, H1, W1), ..., (B, HN, WN)]

    Return:
    :tensor_flatten: (B, L, C), should be [(B, L1, C), ..., (B, LN, C)]
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
        # print(tensor.shape, new_tensor.shape)
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