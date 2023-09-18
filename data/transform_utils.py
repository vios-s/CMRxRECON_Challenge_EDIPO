import torch
import numpy as np
import torch.nn.functional as F
import torchvision

from typing import Union, Tuple

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)

def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std

def pad_size_tensor(data: torch.Tensor, mask: torch.Tensor, desired_shape: Tuple[int, int]= (512, 256)): 
    # shape: [batch, width, height, t, channels]
    w_, h_ = desired_shape #desired shapes
    h, w = data.shape[-3], data.shape[-4] #actual original shapes #TODO dims -3 and -4
    h_pad, w_pad = (h_-h), (w_-w)
    
    if data.dim() == 4:
        pad = (0, 0,
               0, 0,
                h_pad//2, h_pad//2,
                w_pad//2, w_pad//2
            )        
    elif data.dim() == 5:
        pad = (0, 0,
               0, 0,
               0, 0,
                h_pad // 2, h_pad // 2,
                w_pad // 2, w_pad // 2,
            )

    pad_4_mask = (0, 0,
                h_pad // 2, h_pad // 2,
                w_pad // 2, w_pad // 2,
                
                )
    
    data_padded = F.pad(data, pad, mode='constant', value=0)
    mask_padded = F.pad(mask, pad_4_mask, mode='constant', value=0)
    
    # print("pad_size_tensor data_padded: ", data_padded.size())
    # print("pad_size_tensor mask_padded: ", mask_padded.size())
    
    return data_padded, mask_padded

def crop_to_depad(data, metadata):
    
    ori_height, ori_width = metadata['height'], metadata['width']    
    # print(ori_height, ori_width)
    data = data.permute(0, 3, 4, 2, 1)
    w_crop = (data.shape[-1] - ori_width) // 2
    h_crop = (data.shape[-2] - ori_height) // 2
    
    data = torchvision.transforms.functional.crop(data, h_crop, w_crop, ori_height, ori_width)    

    return data.permute(0, 4, 3, 1, 2)

def new_crop(data, metadata):
    ori_height, ori_width = metadata['height'], metadata['width']
    data = data.permute(0, 1, 3, 2)
    w_crop = (data.shape[-1] - ori_width) // 2
    h_crop = (data.shape[-2] - ori_height) // 2

    data = torchvision.transforms.functional.crop(data, h_crop, w_crop, ori_height, ori_width)

    return data.permute(0, 1, 3, 2)
