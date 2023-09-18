import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Callable

import numpy as np

import sys
sys.path.append("..")
from utils.fft import *
from models import Unet

class CineNet(nn.Module):
    def __init__(
        self,
        num_cascades: int,
        chans: int,
        pools: int,
        dynamic_type: str,
        weight_sharing: bool,
        datalayer: nn.Module,
        save_space: bool,
        reset_cache: bool
    ):

        super().__init__()

        self.num_cascades = num_cascades

        if dynamic_type in ['XF', 'XT']:
            if weight_sharing:
                self.model = Unet(chans, pools, dims=2)
            else:
                self.model = nn.ModuleList(
                    [Unet(chans, pools, dims=2), Unet(chans, pools, dims=2)])
        elif dynamic_type == '3D':
            self.model = Unet(chans, pools, dims=3)
        else:
            self.model = Unet(chans, pools, dims=2)

        self.gradR = torch.nn.ModuleList(
            [CineNetBlock(self.model, dynamic_type, weight_sharing)
                for _ in range(self.num_cascades)]
        )
        self.gradD = torch.nn.ModuleList(
            [datalayer for _ in range(self.num_cascades)]
        )

        self.save_space = save_space
        if self.save_space:
            self.forward = self.forward_save_space

        self.reset_cache = reset_cache

    def forward(self, x, y, mask):
        """
        Args:
            x: Input image, shape (b, w, h, t, ch)
            y: Input k-space, shape (b, w, h, t, ch)
            mask: Input mask, shape (b, w, h, ch)
        Returns:
            x: Reconstructed image, shape (b, t, w, h, ch)
        """
        
        x = x.float()  # [b, w, h, t, ch]
        y = y.float()  # [b, t, w, h, ch]
        mask = mask.unsqueeze(1).float()  # [b, 1, w, h, 1]
        x_all = [x]
        x_half_all = []
        for i in range(self.num_cascades):
            x_thalf = (x - self.gradR[i % self.num_cascades](x)) #[b, t, w, h, ch]
            x_half_all.append(x_thalf) #[b, t, w, h, ch]
            x = self.gradD[i % self.num_cascades](x_thalf, y, mask) #[b, w, h, t, ch]
            x_all.append(x) 

        return x_all[-1]

    def forward_save_space(self, x, y, mask):
        """
        Args:
            x: Input image, shape (b, w, h, t, ch)
            y: Input k-space, shape (b, w, h, t, ch)
            mask: Input mask, shape (b, w, h, ch)
        Returns:
            x: Reconstructed image, shape (b, t, w, h, ch)
        """
        x = x.float()  # [b, w, h, t, ch]
        y = y.float()  # [b, w, h, t, ch]
        mask = mask.unsqueeze(3).float()  # [b, w, h, t, ch]

        for i in range(self.num_cascades):

            x_thalf = (x - self.gradR[i % self.num_cascades](x)) #[b, w, h, t, ch]
            x = self.gradD[i % self.num_cascades](x_thalf, y, mask) #[b, w, h, t, ch]

            if self.reset_cache:
                torch.cuda.empty_cache()
                torch.backends.cuda.cufft_plan_cache.clear()

        return x


class CineNetBlock(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        dynamic_type: str,
        weight_sharing: bool
    ):
        super().__init__()

        self.model = model
        self.dynamic_type = dynamic_type
        self.weight_sharing = weight_sharing

    def xfyf_transform(self, image_combined: torch.Tensor) -> torch.Tensor:
        """
        Separate input into two volumes in the rotated planes x-f and y-f
        (or x-t, y-t if in 'XT' dynamics mode). After being processed by
        their respective U-Nets, the volumes are then combined back into one.
        """
        # b, t, w, h, ch = image_combined.shape
        # e.g., torch.Size([1, 256, 512, 12, 2])
        b, h, w, t, ch = image_combined.shape

        # Subtract the image temporal average for numerical stability
        image_temp = image_combined.clone()
        image_mean = torch.stack(
            t * [torch.mean(image_temp, dim=-2)], dim=-2)  # TODO

        x = image_combined - image_mean
        # image_mean torch.Size([1, 256, 512, 12, 2])

        if self.dynamic_type == 'XF':
            # Apply temporal FFT
            x = fft1c(x)
            x = x.permute(0, 3, 1, 2, 4)  # b,t,h,w,2

        # Reshape to xf, yf planes
        xf = x.clone().permute(0, 2, 4, 3, 1).reshape(
            b*h, 2, w, t)  # [b, h, ch, w, t] -> [b*h, ch, w, t]
        yf = x.clone().permute(0, 3, 4, 2, 1).reshape(
            b*w, 2, h, t)  # [b, w, ch, h, t] -> [b*w, ch, h, t]

        # UNet opearting on temporal transformed xf, yf-domain
        if self.weight_sharing:
            xf = self.model(xf)
            yf = self.model(yf)
        else:
            model_xf, model_yf = self.model
            xf = model_xf(xf)
            yf = model_yf(yf)

        # Reshape from xf, yf
        xf_r = xf.view(b, h, 1, 2, w, t).permute(
            0, 5, 2, 1, 4, 3)  # b,t,1,h,w,2
        yf_r = yf.view(b, w, 1, 2, h, t).permute(
            0, 5, 2, 4, 1, 3)  # b,t,1,h,w,2

        out = 0.5 * (xf_r + yf_r)

        if self.dynamic_type == 'XF':
            # Apply temporal IFFT
            out = out.permute(0, 2, 3, 4, 1, 5)  # b,1,h,w,t,2
            out = ifft1c(out)
            out = out.permute(0, 4, 1, 2, 3, 5)  # b,t,1,h,w,2

        # Residual connection
        # print("out", out.size())
        # out torch.Size([b, 12, 1, 256, 512, 2])
        # print("image_mean", image_mean.size())
        # image_mean torch.Size([b, 256, 512, 12, 2])
        # print("image_mean.unsqueeze(2)", image_mean.unsqueeze(2).size())
        # image_mean.unsqueeze(2) torch.Size([b, 256, 1, 512, 12, 2])
        image_mean = image_mean.permute(0, 3, 1, 2, 4).unsqueeze(2)
        out += image_mean

        # print("out", out.size()) #out torch.Size([b, 12, 256, 512, 2])
        return out.squeeze(2).permute(0, 2, 3, 1, 4)

    def forward(
        self,
        image_pred: torch.Tensor
    ) -> torch.Tensor:

        # b, w, h, t, ch = image_pred.shape #e.g., torch.Size([b, 256, 512, 12, 2])
        # e.g., torch.Size([b, 256, 512, 12, 2]) #TODO W and H inverted
        b, h, w, t, ch = image_pred.shape
        x = image_pred.clone()

        if self.dynamic_type in ['XF', 'XT']:
            # model_out torch.Size([b, 12, 256, 512, 2])
            model_out = self.xfyf_transform(x)  # [b, t, h, w, ch]

        elif self.dynamic_type == '2D':
            # Batch dimension b=1. Make first dimension time so
            # that each slice is trained independently. This is
            # similar to static MRI reconstruction.

            # Input to model has shape (t, ch, h, w)
            image_in = image_pred.permute(0, 3, 4, 2, 1).reshape(b*t, ch, h, w)
            model_out = self.model(image_in).reshape(
                b, t, h, w, ch)  # [b, t, h, w, ch]

        elif self.dynamic_type == '3D':
            # In this mode the whole spatio-temporal volume is
            # processed by a 3D U-Net at once.

            # Input to model has shape (b, ch, t, h, w)
            image_in = image_pred.permute(
                0, 4, 3, 2, 1).reshape(b, ch, t, h, w)
            model_out = self.model(image_in).reshape(
                b, t, h, w, ch)  # [b, t, h, w, ch]

        else:
            raise ValueError(f"Unknown dynamic type {self.dynamic_type}")

        return model_out # [b, t, h, w, ch]
