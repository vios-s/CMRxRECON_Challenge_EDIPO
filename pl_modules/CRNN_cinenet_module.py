from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import sys

sys.path.append('..')
from .mri_module import MriModule
from utils.math import complex_abs
from utils.losses import SSIMLoss, PerpLoss, PerpLossSSIM, HighPassLoss
from models import CineNet_RNN,CineNet_RNN_NWS, CineNet_RNN_diff, DCLayer
from data.transform_utils import crop_to_depad


class CRNN_CineNetModule(MriModule):
    def __init__(
            self,
            num_cascades: int = 4,
            chans: int = 18,
            pools: int = 3,
            dynamic_type: str = 'CRNN',
            weight_sharing: bool = True,
            data_term: str = 'DC',
            lambda_: float = np.log(np.exp(1) - 1.) / 1.,
            learnable: bool = True,
            lr: float = 3e-4,
            lr_step_size: int = 40,
            lr_gamma: float = 0.1,
            weight_decay: float = 0.0,
            save_space: bool = False,
            reset_cache: bool = False,
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
        self.highpassloss = HighPassLoss()
        assert self.dynamic_type in ['CRNN', 'CRNN_diff'], \
            "dynamic_type argument must be one of 'CRNN_diff' or'CRNN'"

        if self.dynamic_type == 'CRNN':
            if self.weight_sharing is True:
                self.cinenet = CineNet_RNN(
                    num_cascades=self.num_cascades,
                    chans=self.chans,
                    datalayer=self.datalayer,
                )
            else:
                self.cinenet = CineNet_RNN_NWS(
                    num_cascades=self.num_cascades,
                    chans=self.chans,
                    datalayer=self.datalayer,
                )
        elif self.dynamic_type == 'CRNN_diff':
            self.cinenet = CineNet_RNN_diff(
                num_cascades=self.num_cascades,
                chans=self.chans,
                datalayer=self.datalayer,
            )
        else:
            raise NotImplementedError(f"Dynamic_type {self.dynamic_type} not implemented")

    def forward(self, image, kspace, mask, acc):
        return self.cinenet(image, kspace, mask, acc)

    @staticmethod
    def _postprocess(data, mean, std):
        mean = mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        return complex_abs(data * std + mean)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        output = self(batch.image, batch.kspace, batch.mask, batch.acc)

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        parser.add_argument("--num_cascades", default=5, type=int, help="Number of iterations")
        parser.add_argument("--chans", default=48, type=int, help="Number of channels in CineNet blocks")
        parser.add_argument("--pools", default=3, type=int, help="Number of U-Net pooling for U-Net in CineNet blocks")
        parser.add_argument("--dynamic_type", default='CRNN',
                            choices=['CRNN', 'CRNN_diff'], type=str,
                            help="Architectural variation for dynamic reconstruction. Options are ['CRNN_diff','CRNN']")
        parser.add_argument("--weight_sharing", default=False, type=bool,
                            help="Allows parameter sharing of U-Nets in x-f, y-f planes")
        parser.add_argument("--data_term", default='DC', choices=['DC', 'GD', 'PG', 'VS'], type=str,
                            help="Data consistency term to use. Options are ['DC', 'GD', 'PG', 'VS']")
        parser.add_argument("--lambda_", default=0.5, type=float, help="np.log(np.exp(1) - 1.) / 1.Init value of data consistency block (DCB)")
        parser.add_argument("--learnable", default=True, type=bool, help="Whether to learn lambda_")
        parser.add_argument("--lr", default=2e-4, type=float, help="Adam learning rate")
        parser.add_argument("--lr_step_size", default=40, type=int, help="Epoch at which to decrease step size")
        parser.add_argument("--lr_gamma", default=0.1, type=float, help="Extent to which step size should be decreased")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Strength of weight decay regularization")
        parser.add_argument("--save_space", type=bool, default=True)
        parser.add_argument("--reset_cache", type=bool, default=False)
        return parser