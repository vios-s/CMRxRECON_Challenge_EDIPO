from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any

import torch
import numpy as np
import pandas as pd
from lightning.pytorch import LightningModule
from torchmetrics.metric import Metric

import sys
sys.path.append('../')
from utils import io

class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity
    
class MriModule(LightningModule):
    def __init__(
        self,
        num_log_images: int=16,
        recon_dir = "/output"
    ):
        super().__init__()
        
        self.num_log_images = num_log_images
        self.recon_dir = recon_dir
        self.val_log_indices = None
        self.train_log_indices = [0]
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.TrainLoss = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TestLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()
        
    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, self.global_step)
    
    def on_validation_epoch_end(self):
        pass
        
    def on_test_epoch_end(self):
        
        outputs = defaultdict(dict)

        # use dicts for aggregation to handle duplicate slices in ddp mode
        for log in self.test_step_outputs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice_num"])):
                outputs[fname][int(slice_num.cpu())] = log["output"][i]

        # stack all the slices for each file
        for fname in outputs:
            outputs[fname] = np.stack(
                [out for _, out in sorted(outputs[fname].items())]
            )

        # pull the default_root_dir if we have a trainer, otherwise save to cwd
        save_path = Path.cwd() / self.recon_dir
        self.print(f"Saving reconstructions to {save_path}")
        
        io.save_reconstructions(outputs, save_path)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_log_images", type=int, default=16, help="Number of images to log")
        parser.add_argument("--recon_dir", default="/output", type=str)
        
        return parser
