# CMRxRECON Challenge -- EDIPO Group

## Introduction

This is the repository for the inferece code from EDIPO group for the CMRxRECON challenge. The code is based on the [fastMRI](https://github.com/facebookresearch/fastMRI) with Pytorch Lightning 2.0 framework. 

The file structure is as follows:

```
├── data
│   ├── __init__.py   
│   ├── mri_data.py                             # How to read data and arrange dataset
│   ├── transforms.py                           # Data transforms
│   └── transform_utils.py                      # Data transforms helper functions
├── inference.py                                # The main inference script
├── input                                       # Input folder                                 
│   ├── AccFactor04                             # Acceleration factor 4
│   ├── AccFactor08                             # Acceleration factor 8
│   └── AccFactor10                             # Acceleration factor 10
├── logs
│   └── Experiment1
│       └── checkpoints
│           └── epoch=x-step=xxxx.ckpt          # Checkpoint file
├── models
│   ├── cinenet.py                              # CineNet model
│   ├── datalayer.py                            # Data consistency layers
│   ├── __init__.py
│   ├── recurrent_cinenet_diffcas.py            # Recurrent CineNet model with different cascades
│   ├── recurrent_cinenet_no_weight_sharing.py  # Recurrent CineNet model without weight sharing
│   ├── recurrent_cinenet.py                    # Recurrent CineNet model
│   └── unet.py                                 # U-Net model
├── output                                      # Output folder
├── pl_modules                                  # Pytorch Lightning modules
│   ├── cinenet_module.py                       # CineNet model pl_module
│   ├── CRNN_cinenet_module.py                  # Recurrent CineNet model pl_module
│   ├── data_module.py                          # Data module
│   ├── __init__.py
│   └── mri_module.py                           # MRI module, parent of all other network pl_modules
├── README.md
├── requirements.txt                            # Need to pip install this file
├── utils                                       # Utility functions                                    
│    ├── evaluate.py                            # Evaluation functions                           
│    ├── fft.py                                 # FFT functions                                       
│    ├── __init__.py
│    ├── io.py                                  # IO functions, save reconstructions to mat                                    
│    ├── losses.py                              # Loss functions
│    └── math.py                                # Complex math functions
└── visualize_and_evaluate.py                  # Code to make some visualization of recon images and compute SSIM-NMSE-PSNR metrics
```

## How to run

First install the requirement as follows:

```bash
pip install -r requirements.txt
```

Then run the inference script as follows:

```bash
python inference.py --checkpoint_path logs/Experiment1/checkpoints/epoch=x-step=xxxx.ckpt 
```

Note the input should be located in `input` folder, and the reconstructions will be saved in the `output` folder.

## Results


## References

[1]. 
