import os
from pathlib import Path
from argparse import ArgumentParser

import torch
import lightning.pytorch as pl
from data.transforms import CineNetDataTransform
from pl_modules import MriDataModule, CineNetModule, CRNN_CineNetModule, CRNN_SR_CineNetModule

torch.set_float32_matmul_precision('medium')

def build_args():
    parser = ArgumentParser()
    exp_name = "crnn_NWS_6c_64chan_L1_ssim_HI_SR"
    default_log_path = Path("logs") / exp_name

    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    parser.add_argument("--exp_name", default=exp_name, type=str)
    parser.add_argument("--mode", default="test", type=str, choices=["train", "test"])
    parser.add_argument("--model", default="crnn_sr", type=str, choices=["cinenet", "crnn", "crnn_sr"])
    parser.add_argument("--ckpt_path", default=None, type=str)

    parser = MriDataModule.add_data_specific_args(parser)
    if parser.parse_known_args()[0].model == "cinenet":
        parser = CineNetModule.add_model_specific_args(parser)
    elif parser.parse_known_args()[0].model == "crnn":
        parser = CRNN_CineNetModule.add_model_specific_args(parser)
    elif parser.parse_known_args()[0].model == "crnn_sr":
        parser = CRNN_SR_CineNetModule.add_model_specific_args(parser)


        
    parser.set_defaults(
        seed=42,
        batch_size=1,
        default_root_dir=default_log_path,
        time_window=12        
    )
    
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    print("Input data store in:", input_dir)
    print("Output data store in:", output_dir)

    # checkpoints
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
        
    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            verbose=True,
        )
    ]
    
    if args.ckpt_path is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.ckpt_path = ckpt_list[-1]
            
    print(args.ckpt_path)
    
    return args

def main():
    args = build_args()
    pl.seed_everything(args.seed)
    
    #* Data Module
    test_transform = CineNetDataTransform(use_seed=False, time_window=args.time_window)
    
    #* Data Loader
    data_module = MriDataModule(
        data_path=Path(args.input),
        test_transform=test_transform,
        test_sample_rate=args.test_sample_rate,
        use_dataset_cache=args.use_dataset_cache,
        batch_size=args.batch_size,
        num_workers=args.num_workers, #os.cpu_count()
        distributed_sampler=False

    )
    
    #* Network Model
    if args.model == "cinenet":
        model = CineNetModule(
            num_cascades=args.num_cascades,
            chans=args.chans,
            pools=args.pools,
            dynamic_type=args.dynamic_type,
            weight_sharing=args.weight_sharing,
            data_term=args.data_term,
            lambda_=args.lambda_,
            learnable=args.learnable,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            save_space=args.save_space,
            reset_cache=args.reset_cache,
        )
    elif args.model == "crnn":
        model = CRNN_CineNetModule(
        num_cascades=args.num_cascades,
        chans=args.chans,
        pools=args.pools,
        dynamic_type=args.dynamic_type,
        weight_sharing=args.weight_sharing,
        data_term=args.data_term,
        lambda_=args.lambda_,
        learnable=args.learnable,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        save_space=args.save_space,
        reset_cache=args.reset_cache,
    )
    elif args.model == "crnn_sr":
        model = CRNN_SR_CineNetModule(
        num_cascades=args.num_cascades,
        chans=args.chans,
        pools=args.pools,
        dynamic_type=args.dynamic_type,
        weight_sharing=args.weight_sharing,
        data_term=args.data_term,
        lambda_=args.lambda_,
        learnable=args.learnable,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        save_space=args.save_space,
        reset_cache=args.reset_cache,
    )

        
    print("Done Loading Data and Model...")

    #* Trainer
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        accelerator="gpu",
        logger=False,
        callbacks=args.callbacks,
        default_root_dir=args.default_root_dir,
    )
    
    #* Test
    if args.mode == 'test':
        print("Testing "
            f"{(args.model).upper()} with "
            f"{args.num_cascades} unrolled iterations.\n")
        trainer.test(model, data_module, ckpt_path=args.ckpt_path)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    
    
if __name__ == '__main__':
    main()
