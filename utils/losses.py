import torch
import torch.nn as nn
import torch.nn.functional as F

from .math import complex_abs


class SSIM(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, Xt: torch.Tensor, Yt: torch.Tensor, data_range=None, full=False):
        assert isinstance(self.w, torch.Tensor)
        Xt = (Xt / Xt.max()).unsqueeze(2)
        Yt = (Yt / Yt.max()).unsqueeze(2)
        ssims = 0.0
        for t in range(Xt.shape[1]):

            X = Xt[:, t, :, :, :].permute(0, 1, 3, 2)
            Y = Yt[:, t, :, :, :].permute(0, 1, 3, 2)

            if data_range is None:
                data_range = torch.ones_like(Y)  # * Y.max()
                p = (self.win_size - 1) // 2
                data_range = data_range[:, :, p:-p, p:-p]
            data_range = data_range[:, None, None, None]
            C1 = (self.k1 * data_range) ** 2
            C2 = (self.k2 * data_range) ** 2
            ux = F.conv2d(X, self.w)  # typing: ignore
            uy = F.conv2d(Y, self.w)  #
            uxx = F.conv2d(X * X, self.w)
            uyy = F.conv2d(Y * Y, self.w)
            uxy = F.conv2d(X * Y, self.w)
            vx = self.cov_norm * (uxx - ux * ux)
            vy = self.cov_norm * (uyy - uy * uy)
            vxy = self.cov_norm * (uxy - ux * uy)
            A1, A2, B1, B2 = (
                2 * ux * uy + C1,
                2 * vxy + C2,
                ux ** 2 + uy ** 2 + C1,
                vx + vy + C2,
            )
            D = B1 * B2
            S = (A1 * A2) / D

            if full:
                ssims += 1 - S
            else:
                ssims += 1 - S.mean()

        return ssims / Xt.shape[1]


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    @staticmethod
    def _postprocess(data, mean, std):
        mean = mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        return complex_abs(data * std + mean)

    def forward(self, Xt: torch.Tensor, Yt: torch.Tensor, mean, std, data_range=None, full=False):
        assert isinstance(self.w, torch.Tensor)

        ssims = 0.0

        if Xt.is_complex():
            Xt = torch.view_as_real(Xt)
        if Yt.is_complex():
            Yt = torch.view_as_real(Yt)

        Xt = self._postprocess(Xt, mean, std)
        Xt = (Xt / Xt.max()).unsqueeze(1)
        Yt = self._postprocess(Yt, mean, std)
        Yt = (Yt / Yt.max()).unsqueeze(1)

        for t in range(Xt.shape[-1]):

            X = Xt[:, :, :, :, t].permute(0, 1, 3, 2)
            Y = Yt[:, :, :, :, t].permute(0, 1, 3, 2)

            if data_range is None:
                data_range = torch.ones_like(Y)  # * Y.max()
                p = (self.win_size - 1) // 2
                data_range = data_range[:, :, p:-p, p:-p]
            data_range = data_range[:, None, None, None]
            C1 = (self.k1 * data_range) ** 2
            C2 = (self.k2 * data_range) ** 2
            ux = F.conv2d(X, self.w)  # typing: ignore
            uy = F.conv2d(Y, self.w)  #
            uxx = F.conv2d(X * X, self.w)
            uyy = F.conv2d(Y * Y, self.w)
            uxy = F.conv2d(X * Y, self.w)
            vx = self.cov_norm * (uxx - ux * ux)
            vy = self.cov_norm * (uyy - uy * uy)
            vxy = self.cov_norm * (uxy - ux * uy)
            A1, A2, B1, B2 = (
                2 * ux * uy + C1,
                2 * vxy + C2,
                ux ** 2 + uy ** 2 + C1,
                vx + vy + C2,
            )
            D = B1 * B2
            S = (A1 * A2) / D

            if full:
                ssims += 1 - S
            else:
                ssims += 1 - S.mean()

        return ssims / Xt.shape[-1]


class PerpLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        X = torch.view_as_complex(X)
        Y = torch.view_as_complex(Y)

        assert X.is_complex()
        assert Y.is_complex()

        mag_input = torch.abs(X)
        mag_target = torch.abs(Y)
        cross = torch.abs(X.real * Y.imag - X.imag * Y.real)

        angle = torch.atan2(X.imag, X.real) - torch.atan2(Y.imag, Y.real)
        ploss = torch.abs(cross) / (mag_input + 1e-8)

        aligned_mask = (torch.cos(angle) < 0).bool()

        final_term = torch.zeros_like(ploss)
        final_term[aligned_mask] = mag_target[aligned_mask] + (mag_target[aligned_mask] - ploss[aligned_mask])
        final_term[~aligned_mask] = ploss[~aligned_mask]
        return (final_term + F.mse_loss(mag_input, mag_target)).mean()


class PerpLossSSIM(nn.Module):
    def __init__(self):
        super().__init__()

        self.ssim = SSIMLoss()
        self.param = nn.Parameter(torch.ones(1) / 2)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, mean, std):
        X = torch.view_as_complex(X)
        Y = torch.view_as_complex(Y)

        assert X.is_complex()
        assert Y.is_complex()

        mag_input = torch.abs(X)
        mag_target = torch.abs(Y)
        cross = torch.abs(X.real * Y.imag - X.imag * Y.real)

        angle = torch.atan2(X.imag, X.real) - torch.atan2(Y.imag, Y.real)
        ploss = torch.abs(cross) / (mag_input + 1e-8)

        aligned_mask = (torch.cos(angle) < 0).bool()

        final_term = torch.zeros_like(ploss)
        final_term[aligned_mask] = mag_target[aligned_mask] + (mag_target[aligned_mask] - ploss[aligned_mask])
        final_term[~aligned_mask] = ploss[~aligned_mask]
        ssim_loss = (self.ssim(X, Y, mean, std)) / mag_input.shape[0]

        return (final_term.mean() * torch.clamp(self.param, 0, 1) + (1 - torch.clamp(self.param, 0, 1)) * ssim_loss)


class HighPassLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = torch.nn.L1Loss()

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        high_pass_loss = self.l1_loss(self.high_pass_filter(X), self.high_pass_filter(Y))

        return high_pass_loss

    def high_pass_filter(self, images):
        # Define the high pass filter kernel
        kernel = torch.tensor([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(images.device)

        filtered_images = torch.zeros_like(images)

        # Iterate over the batch, time, and channel dimensions
        for b in range(images.size(0)):
            for t in range(images.size(3)):
                for c in range(images.size(4)):
                    # Apply high pass filter using convolution
                    filtered_image = F.conv2d(images[b, :, :, t, c].unsqueeze(0).unsqueeze(0), kernel, padding=1)
                    filtered_images[b, :, :, t, c] = filtered_image.squeeze()

        return filtered_images


class HighPassImageLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = torch.nn.L1Loss()

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        high_pass_loss = self.l1_loss(self.high_pass_filter(X), self.high_pass_filter(Y))

        return high_pass_loss

    def high_pass_filter(self, images):
        # Define the high pass filter kernel
        kernel = torch.tensor([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(images.device)

        filtered_images = torch.zeros_like(images)

        # Iterate over the batch, time, and channel dimensions
        for b in range(images.size(0)):
            for t in range(images.size(1)):
                # Apply high pass filter using convolution
                filtered_image = F.conv2d(images[b, t, ...].unsqueeze(0), kernel, padding=1)
                filtered_images[b, t, ...] = filtered_image.squeeze()

        return filtered_images


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        """
        Total variation loss
        https://github.com/jxgu1016/Total_Variation_Loss.pytorch
        Args:
            tv_loss_weight (int):
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# --------------------------------------------
# Charbonnier loss
# --------------------------------------------
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss