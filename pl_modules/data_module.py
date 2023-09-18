from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional

import torch
import lightning.pytorch as pl
import os
import sys
sys.path.append('..')
from data import SliceDataset


class MriDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        test_transform: Callable,
        test_sample_rate: Optional[float] = None,
        use_dataset_cache: bool = True,
        batch_size: int = 1,
        num_workers: int =4,
        distributed_sampler: bool = False) -> None:
        """
        Args:
            data_path: Path to root data directory (with expected subdirectories
                train/valid/test).
            test_transform: A transform object for the test split.
            sample_rate: Fraction of slices of the training data split to use.
            use_dataset_cache_file: Whether to cache dataset metadata. This is
                very useful for large datasets.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This
                should be set to True if training with ddp.
        """
        super().__init__()
        
        self.data_path = data_path
        self.test_transform = test_transform
        self.test_sample_rate = test_sample_rate
        self.use_dataset_cache = use_dataset_cache
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        
    def prepare_data(self) -> None:
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        if self.use_dataset_cache:
            data_paths = [
                self.data_path
            ]
            
            data_transforms = [
                self.test_transform
            ]
            
            data_sample_rates = [
                self.test_sample_rate
            ]
            
            dataset_cache_files = [
                'TestSet_cache_file.pkl'
            ]
            
            for _, (data_path, data_transform, data_sample_rate, dataset_cache_file) in enumerate(
                zip(data_paths, data_transforms, data_sample_rates, dataset_cache_files)):
                _ = SliceDataset(
                    root=data_path,
                    transform=data_transform,
                    sample_rate=data_sample_rate,
                    use_dataset_cache=self.use_dataset_cache,
                    dataset_cache_file=dataset_cache_file
                )
                print('Dataset cache file created for {}'.format(data_path))
        else:
            print('No dataset cache file created')          
    
    def _create_dataloader(
        self,
        data_transform: Callable,
        data_partition: str,
        sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:
        
        is_train = True if data_partition == 'TrainingSet' else False
        data_path = self.data_path #/ data_partition
        
        dataset = SliceDataset(
            root=data_path,
            transform=data_transform,
            sample_rate=sample_rate,
            use_dataset_cache=self.use_dataset_cache,
            dataset_cache_file=f'{data_partition}_cache_file.pkl'
        )
        
        sampler = None
        
        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
        
        batch_size = 1 if data_partition == 'TestSet' else self.batch_size
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )
        
        return dataloader

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        pass
        
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        pass
        
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self._create_dataloader(
            data_transform=self.test_transform,
            data_partition='TestSet',
            sample_rate=self.test_sample_rate
        )
        
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        parser.add_argument("--data_path", type=Path, default=None, help="Path to the root data directory")
        parser.add_argument("--test_sample_rate", type=float, default=None, help="Fraction of slices to use for testing")
        parser.add_argument("--use_dataset_cache", type=bool, default=True, help="Whether to cache dataset metadata in memory")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
        parser.add_argument("--distributed_sampler", type=bool, default=False, help="Use DDP training.")
        return parser
