from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('..')
from .mri_module import MriModule
from utils import evaluate
from utils.math import complex_abs
from utils.losses import SSIMLoss, PerpLoss, PerpLossSSIM
from models import CineNet, CineNet_RNN, DCLayer
from data.transform_utils import *

class CineNetModule(MriModule):
    def __init__(
            self,
            num_cascades: int = 4,
            chans: int = 18,
            pools: int = 3,
            dynamic_type: str = 'XF',
            weight_sharing: bool = True,
            data_term: str = 'DC',
            lambda_: float = np.log(np.exp(1)-1.)/1.,
            learnable: bool = True,
            lr: float = 3e-4,
            lr_step_size: int = 40,
            lr_gamma: float = 0.1,
            weight_decay: float = 0.0,
            save_space: bool=False,
            reset_cache: bool=False,
            **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.chans = chans
        self.pools = pools
        self.dynamic_type = dynamic_type
        self.weight_sharing = weight_sharing
        self.data_term = data_term
        self.lambda_ = lambda_
        self.learnable = learnable
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.save_space = save_space
        self.reset_cache = reset_cache

        if self.data_term == 'DC':
            self.datalayer = DCLayer(
                lambda_init=self.lambda_,
                learnable=self.learnable
            )
        elif self.data_term == 'GD':
            raise NotImplementedError
        elif self.data_term == 'PG':
            raise NotImplementedError
        elif self.data_term == 'VS':
            raise NotImplementedError
        else:
            raise ValueError(f"Data term {self.data_term} not recognized")

        self.ssim = SSIMLoss()
        self.perp = PerpLoss()
        self.l1_loss = torch.nn.L1Loss()
        self.l2_loss = torch.nn.MSELoss()
        self.perpssim = PerpLossSSIM()

        assert self.dynamic_type in ['XF', 'XT', '2D', '3D', 'CRNN'], \
            "dynamic_type argument must be one of 'XF', 'XT', '2D', '3D' or 'CRNN'"

        if self.dynamic_type == 'CRNN':
            self.cinenet = CineNet_RNN(
                num_cascades=self.num_cascades,
                chans=self.chans,
                datalayer=self.datalayer,
            )
        else:
            self.cinenet = CineNet(
                num_cascades=self.num_cascades,
                chans=self.chans,
                pools=self.pools,
                dynamic_type=self.dynamic_type,
                weight_sharing=self.weight_sharing,
                datalayer=self.datalayer,
                save_space=self.save_space,
                reset_cache=self.reset_cache
                
            )

    def forward(self, image, kspace, mask):
        return self.cinenet(image, kspace, mask)

    @staticmethod
    def _postprocess(data, mean, std):
        mean = mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        return complex_abs(data * std + mean)

    def training_step(self, batch, batch_idx):
        output = self(batch.image, batch.kspace, batch.mask) # [b, h, w, t, ch]
        # print('output ', output.shape, batch.metadata)
        output = crop_to_depad(output, batch.metadata).contiguous() # [b, h, w, t, ch]
        target = crop_to_depad(batch.target, batch.metadata).contiguous() # [b, h, w, t, ch]
        post_output = self._postprocess(output, batch.mean, batch.std) # [b, h, w, t]
        post_target = self._postprocess(target, batch.mean, batch.std) # [b, h, w, t]

        # loss = F.l1_loss(output, target) + 0.1 * self.ssim(
        #     (post_output / post_output.max()).unsqueeze(1),
        #     (post_target / post_target.max()).unsqueeze(1),
        # )
        loss = self.perpssim(output, target, batch.mean, batch.std)

        self.log("TrainLoss", loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch.image, batch.kspace, batch.mask)
        # print('output ', output.shape, batch.metadata)
        output = crop_to_depad(output, batch.metadata)
        image = crop_to_depad(batch.image, batch.metadata)
        target = crop_to_depad(batch.target, batch.metadata)
        
        #validation_step - batch.image:  torch.Size([1, 512, 256, 12, 2])
        #validation_step - output:  torch.Size([1, 512, 256, 12, 2])
        #validation_step - batch.target:  torch.Size([1, 512, 256, 12, 2])

        post_image = self._postprocess(image, batch.mean, batch.std)
        post_output = self._postprocess(output, batch.mean, batch.std)
        post_target = self._postprocess(target, batch.mean, batch.std)

        # validation_step - post_image:  torch.Size([1, 512, 256, 12])
        # validation_step - post_output:  torch.Size([1, 512, 256, 12])
        # validation_step - post_target:  torch.Size([1, 512, 256, 12])

        val_logs = {
            "batch_idx": batch_idx,
            "image": post_image,
            "kspace": batch.kspace,
            "output": post_output,
            "target": post_target,
            "mask": batch.mask,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "val_loss": self.perpssim(output, target, batch.mean, batch.std)
            # "val_loss": self.l2_loss(post_output[..., 0], post_target[..., 0]) + self.l2_loss(post_output[..., 1], post_target[..., 1]) +
            # 0.1 * self.ssim(output, target, batch.mean, batch.std)
            # "val_loss": F.l1_loss(output, target) + 0.1 * self.ssim(
            #     (post_output / post_output.max()).unsqueeze(1),
            #     (post_target / post_target.max()).unsqueeze(1),
            #     )
        }

        for k in ("batch_idx", "image", "kspace", "output", "target", "mask", "fname", "slice_num", "max_value"):
            if k not in val_logs.keys():
                raise ValueError(f"Key {k} not found in val_logs")

        if val_logs["output"].ndim != 4 or val_logs["target"].ndim != 4 or val_logs["image"].ndim != 4:
            raise ValueError(f"Wrong number of dimensions: Output {val_logs['output'].ndim}, \
                                Target {val_logs['target'].ndim}, Image {val_logs['image'].ndim}")

        # * pick an image to log
        if self.val_log_indices is None:
            self.val_log_indices = list(np.random.permutation(
                len(self.trainer.val_dataloaders))[: self.num_log_images])

        # log images to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]

        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"           

                image = val_logs["image"][i][:,:,0].unsqueeze(0)
                target = val_logs["target"][i][:,:,0].unsqueeze(0)
                output = val_logs["output"][i][:,:,0].unsqueeze(0)
                mask = val_logs["mask"][i].permute(2, 0, 1)
                error = torch.abs(target - output)
                image = image / image.max()
                output = output / output.max()
                target = target / target.max()
                error = error / error.max()

                self.log_image(f"{key}/input", image)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/target", target)
                self.log_image(f"{key}/error", error)
                self.log_image(f"{key}/mask", mask)

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval = val_logs["max_value"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output)).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target, output, maxval=maxval)).view(1)
            max_vals[fname] = maxval

        pred = {
            "val_loss": val_logs["val_loss"],
            "mse_vals": mse_vals,
            "target_norms": target_norms,
            "ssim_vals": ssim_vals,
            "max_vals": max_vals
        }
        self.validation_step_outputs.append(pred)
        return pred

    def test_step(self, batch, batch_idx):
        output = self(batch.image, batch.kspace, batch.mask)

        output = crop_to_depad(output, batch.metadata)
        post_output = self._postprocess(output, batch.mean, batch.std)

        test_logs = {
            "batch_idx": batch_idx,
            "output": post_output.detach().cpu(),
            "fname": batch.fname,
            "slice_num": batch.slice_num
        }

        for k in ("batch_idx", "output", "fname", "slice_num"):
            if k not in test_logs.keys():
                raise ValueError(f"Key {k} not found in test_logs")

        if test_logs["output"].ndim != 4:
            raise ValueError(f"Wrong number of dimensions: Output {test_logs['output'].ndim}")

        self.test_step_outputs.append(test_logs)

        return test_logs

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim,
            self.lr_step_size,
            self.lr_gamma
        )

        return [optim], [scheduler]

    # def on_after_backward(self) -> None:
    #     print("on_after_backward enter")
    #     for name, p in self.cinenet.trainable_params:
    #         if p.grad is None:
    #             print(name, p)
    #     print("on_after_backward exit")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)
        parser.add_argument("--num_cascades", default=5, type=int, help="Number of iterations")
        parser.add_argument("--chans", default=16, type=int, help="Number of channels in CineNet blocks")
        parser.add_argument("--pools", default=3, type=int, help="Number of U-Net pooling for U-Net in CineNet blocks")
        parser.add_argument("--dynamic_type", default='XF', choices=['XF', 'XT', '2D', '3D', 'CRNN'], type=str, help="Architectural variation for dynamic reconstruction. Options are ['XF', 'XT', '2D', '3D', 'CRNN']")
        parser.add_argument("--weight_sharing", default=True, type=bool, help="Allows parameter sharing of U-Nets in x-f, y-f planes")
        parser.add_argument("--data_term", default='DC', choices=['DC', 'GD', 'PG', 'VS'], type=str, help="Data consistency term to use. Options are ['DC', 'GD', 'PG', 'VS']")
        parser.add_argument("--lambda_", default=np.log(np.exp(1)-1.)/1., type=float, help="Init value of data consistency block (DCB)")
        parser.add_argument("--learnable", default=True, type=bool, help="Whether to learn lambda_")
        parser.add_argument("--lr", default=1e-4, type=float, help="Adam learning rate")
        parser.add_argument("--lr_step_size", default=140, type=int, help="Epoch at which to decrease step size")
        parser.add_argument("--lr_gamma", default=0.01, type=float, help="Extent to which step size should be decreased")
        parser.add_argument("--weight_decay", default=0.001, type=float, help="Strength of weight decay regularization")
        parser.add_argument("--save_space", type=bool, default=True)
        parser.add_argument("--reset_cache", type=bool, default=False)
        
        return parser
