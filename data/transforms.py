import random
from typing import Dict, Any, NamedTuple, Optional, List, Tuple, Union

import torch
import numpy as np

import sys
sys.path.append('..')
from .transform_utils import *
from utils.fft import *

class CineSample(NamedTuple):
    
    image: torch.Tensor
    # raw_kspace: torch.Tensor
    kspace: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    metadata: Dict[str, Any]
    acc: int
    

class CineNetDataTransform:
    def __init__(
        self,
        use_seed: bool = True,
        time_window: int = 8,
        padding_size: Tuple = (512, 256)):
        
        self.use_seed = use_seed
        self.time_window = time_window
        self.padding_size = padding_size
        
    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
        ):
        #! 1. Convert to tensor
        if target is not None:
            target = to_tensor(target)
        else:
            target = to_tensor(kspace)
        # target shape [w, h, t, ch]        
        max_value = target.max()
        
        input_kspace = to_tensor(kspace)
        #input_kspace shape [w, h, t, ch]
        mask = to_tensor(np.expand_dims(mask, axis=-1))
        # mask shape [w, h, 1]
        #! 2. Convert to image
        target_image = ifftnc(target)
        # target_image shape [w, h, t, ch]
        input_image = ifftnc(input_kspace)
        # input_image shape [w, h, t, ch]
    
        # calculate acc
        slice = mask.squeeze()
        zero_rows_count = torch.count_nonzero(slice[0,:] == 0, dim=0)
        rate = (slice.shape[1] - 24) / (slice.shape[1] - zero_rows_count - 24)
        
        #! 3. Padding
        padded_input, padded_mask = pad_size_tensor(input_image, mask, self.padding_size)
        # print(padded_input.shape, padded_mask.shape)
        # padded_input shape [512, 256, t, ch]
        # padded_mask shape [512, 256, 1]
        padded_target, _ = pad_size_tensor(target_image, mask, self.padding_size)
        # print(padded_target.shape)
        # padded_target shape [512, 256, t, ch]
        
        #! 4. Apply fft to get related k-space
        padded_input_kspace = fftnc(padded_input)
        # padded_input_kspace shape [512, 256, t, ch]
        #! 5. Apply mask
        ## skipped because the data is already masked
        
        #! 6. Apply time window
        seed = None if not self.use_seed else tuple(map(ord, fname))
        start_time = random.Random(seed).randint(0, 12 - self.time_window)
        
        padded_input = padded_input[:, :, start_time:start_time + self.time_window, :]
        padded_input_kspace = padded_input_kspace[:, :, start_time:start_time + self.time_window, :]
        padded_target = padded_target[:, :, start_time:start_time + self.time_window, :]
        
        #! 7. normalization
        padded_norm_input, mean, std = normalize_instance(padded_input, eps=1e-11)
        padded_norm_target = normalize(padded_target, mean, std, eps=1e-11)
        
        #! 8. update metadata
        attrs.update(
            {
                "padding_size": self.padding_size,
                }
            )
        
        return CineSample(
            image = padded_norm_input,
            # raw_kspace = input_kspace,
            kspace = padded_input_kspace,
            mask = padded_mask,
            target = padded_norm_target,
            fname = fname,
            slice_num = slice_num,
            mean = mean,
            std = std,
            max_value=max_value,
            metadata=attrs,
            acc=int(torch.round(rate)),
        )