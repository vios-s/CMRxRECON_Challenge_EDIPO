# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:54:31 2023

@author: gianlucarloni and evapa
"""

import os

import mat73
from matplotlib import pyplot as plt
import numpy as np
from scipy import io


# De-comment Part 1 or Part 2.

#%% PART 1: VISUALIZATION OF SOME EXAMPLES. MOVE TO PART 2 BELOW FOR COMPUTATION OF THE METRICS (SSIM,PSNR,MSE)

#Define
partition = "TestSet" #TestSet, ValidationSet, TrainingSet
model = "output_3c_TestSet_npy"
accfactor = "AccFactor10" #AccFactor08, AccFactor10, AccFactor04
patient_ID = "P120"
my_file = "cine_sax.mat.npy" # Reconstruction can be in NUPY or MAT format depending on the choice made during inferece (utils/io.py)

if my_file.startswith("cine_lax"):
    WHICH_AXIS = "lax"
else:
    WHICH_AXIS = "sax"
    

# Reconstruction can be in NUPY or MAT format depending on the choice made during inferece (utils/io.py)
# In this case I have all the reconstructions in NPY format, so I need to upload them with np.load() instead of scipy.io.loadmat()
data_mat_recon = rf'Y:/raid/home/gianlucacarloni/CMRxRecon/recon_images_cinenet_models/{partition}/{model}/{accfactor}/{patient_ID}/{my_file}' #TODO: CUSTOMIZE root and subfolders as needed

# Instead, undersampled and fully sampled data were given in MAT format originally
data_mat_original = rf'Y:/raid/home/gianlucacarloni/CMRxRecon/SingleCoil/Cine/{partition}/{accfactor}/{patient_ID}/cine_{WHICH_AXIS}.mat' #CUSTOMIZE...
data_mat_full = rf'Y:/raid/home/gianlucacarloni/CMRxRecon/SingleCoil/Cine/{partition}/FullSample/{patient_ID}/cine_{WHICH_AXIS}.mat' #CUSTOMIZE...
   
  
# dict_recon = io.loadmat(data_mat_recon)
# img_recon = dict_recon["img4ranking"] #Version with matlab
img_recon = np.load(data_mat_recon) #Version with npy

dict_original = mat73.loadmat(data_mat_original)
name_of_undersampled_kspace_variable=""
if accfactor=="AccFactor04":
    name_of_undersampled_kspace_variable = "kspace_single_sub04"
elif accfactor=="AccFactor08":
    name_of_undersampled_kspace_variable = "kspace_single_sub08"
elif accfactor=="AccFactor10":
        name_of_undersampled_kspace_variable = "kspace_single_sub10" 
kspace = dict_original[name_of_undersampled_kspace_variable]
dict_full = mat73.loadmat(data_mat_full)
kspace_full = dict_full["kspace_single_full"]

#img_original = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace))) # 3D COMPUTATION INDUCES SOME ISSUES, LET'S DO IT IN 2D
img_original = np.zeros_like(kspace, dtype=np.float32)
for space in range(kspace.shape[2]):
    for time in range(kspace.shape[3]):
        img_original[:,:,space,time]=np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace[:,:,space,time]))))
        
#img_full = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace_full)))
img_full = np.zeros_like(kspace_full, dtype=np.float32)
for space in range(kspace_full.shape[2]):
    for time in range(kspace_full.shape[3]):
        img_full[:,:,space,time]=np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_full[:,:,space,time]))))
        

# fig = plt.figure()
# fig.tight_layout()

# for i in range(3):
#     plt.subplot(1,3,i+1)
#     plt.imshow(img_recon[:,:,i,0],cmap="gray")
#     plt.axis('off')
#     plt.title("REC-s"+str(i))

# fig = plt.figure()
# fig.tight_layout()
# for i in range(3):
#     plt.subplot(1,3,i+1)
#     plt.imshow(img_original[:,:,i,0],cmap="gray")
#     plt.axis('off')
#     plt.title("Acc04-s"+str(i))


# fig = plt.figure()
# fig.tight_layout()
# for i in range(3):
#     plt.subplot(1,3,i+1)
#     plt.imshow(img_full[:,:,i,0],cmap="gray")
#     plt.axis('off')
#     plt.title("FULL-s"+str(i))


row_img_original = img_original[:,:,0,0]
for i in range(1,3):
    row_img_original = np.concatenate((row_img_original, img_original[:,:,i,0]), axis=1)
plt.figure()
plt.imshow(row_img_original, cmap="gray")  
plt.title(f"{model} - {accfactor}: {patient_ID} {my_file}\nOriginal input: slice 1, slice 2, slice 3") 
    

row_img_recon = img_recon[:,:,0,0]
for i in range(1,3):
    row_img_recon = np.concatenate((row_img_recon, img_recon[:,:,i,0]), axis=1)  
plt.figure()
plt.imshow(row_img_recon, cmap="gray")   
plt.title(f"{model} - {accfactor}: {patient_ID} {my_file}\nReconstruction: slice 1, slice 2, slice 3") 


row_img_full = img_full[:,:,0,0]
for i in range(1,3):
    row_img_full = np.concatenate((row_img_full, img_full[:,:,i,0]), axis=1)
plt.figure()
plt.imshow(row_img_full, cmap="gray")
plt.title(f"{model} - {accfactor}: {patient_ID} {my_file}\n Ground Truth: slice 1, slice 2, slice 3") 
   


#%% PART 2: EVALUATION METRICS
'''
This takes as input the folder of reconstructed images (from inference.py) saved in npy, and computes the metrics CSV file for each
'''
import time
import pandas as pd
from tqdm import tqdm


#option with utility function:
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

def compute_metrics_2D(true_volume, recon_volume):
    #volumes are of shape [H, W, spatial_slices, temporal_instances]
    S = true_volume.shape[2]
    T = true_volume.shape[3]
    
    psnr_val = 0.0
    ssim_val = 0.0
    mse_val = 0.0
    
    for s_space in range(S):
        for t_time in range(T):
            img_full = true_volume[:,:,s_space,t_time].astype(float)
            img_recon = recon_volume[:,:,s_space,t_time].astype(float)

            # Remember, we need to normalize intensities of the images to make values consistent before computing the metrics
            img_full = (img_full-img_full.min())/(img_full.max()-img_full.min())
            img_recon = (img_recon-img_recon.min())/(img_recon.max()-img_recon.min())

            psnr_val += psnr(img_full, img_recon, data_range=1.0)  
            ssim_val += ssim(img_full, img_recon, data_range=1.0)
            mse_val += mse(img_full, img_recon)

    S=float(S)
    T=float(T)
    return psnr_val/(S*T), ssim_val/(S*T), mse_val/(S*T)


##TODO customise the following:
name_of_reconstrucedImages_folder = "output_6c_TrainingSet_npy" #"output_6c_TrainingSet_npy" #E.g., "output_3c", "output_6c"  
name_of_partition = "TrainingSet" ##"TestSet", "ValidationSet", "TrainingSet"
##

df = pd.DataFrame(columns=['Acc Factor', 'Patient name', 'Mat file', 'PSNR', 'SSIM', 'MSE'])
root_path_reconstructed_images = os.path.join(os.getcwd(), "recon_images_cinenet_models", name_of_partition, name_of_reconstrucedImages_folder) #e.g., Y:/raid/home/gianlucacarloni/CMRxRecon/output/
root_path_dataset = os.path.join(os.getcwd(),"SingleCoil","Cine", name_of_partition) #e.g., Y:/raid/home/gianlucacarloni/CMRxRecon/SingleCoil/Cine/TestSet/

acc_factors = os.listdir(root_path_reconstructed_images)

k=0
for acc_factor in acc_factors:
    patients_names = os.listdir(os.path.join(root_path_reconstructed_images,acc_factor))

    for patient_name in tqdm(patients_names):        
        mat_files = os.listdir(os.path.join(root_path_reconstructed_images,acc_factor, patient_name))

        for mat_file in mat_files:
            recon_file_path = os.path.join(root_path_reconstructed_images, acc_factor, patient_name, mat_file)
            
            ## Option 1: If reconstructed images are in MAT 5.0 format
            # dict_recon = io.loadmat(recon_file_path)
            # img_recon = dict_recon["img4ranking"] #the dictionary stores the images directly
            ## TODO: Option 2: Variant whne recon images are in NUMPY format
            img_recon = np.load(recon_file_path)
        
            ## Retrieve the corresponding original fully sampled image, which is in MAT format
            fully_file_path = os.path.join(root_path_dataset, "FullSample", patient_name, mat_file)  
            if fully_file_path.endswith(".npy"):
                fully_file_path = fully_file_path[:-4]
            dict_full = mat73.loadmat(fully_file_path)
            kspace_full = dict_full["kspace_single_full"] #here, instead, we obtain kspaces so we need to convert to image space

            # let's do it slice by slice (2D)
            img_full = np.zeros_like(kspace_full, dtype=np.float32)
            for s_space in range(kspace_full.shape[2]):
                for t_time in range(kspace_full.shape[3]):
                    img_full[:,:,s_space,t_time]=np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_full[:,:,s_space,t_time]))))
                    
            ## Optionally visualize some of the paired images:                            
            # fig = plt.figure()
            # fig.tight_layout()            
            # plt.subplot(121)
            # plt.imshow(img_recon[:,:,1,1],cmap="gray")
            # plt.axis('off')
            # plt.title("REC-1")
            # plt.subplot(122)
            # plt.imshow(img_full[:,:,1,1],cmap="gray")
            # plt.axis('off')
            # plt.title("FULL-1")
            # plt.suptitle(f"{name_of_reconstrucedImages_folder} on {name_of_partition}: {acc_factor} {patient_name} {mat_file}")
            # # plt.savefig("myplot.png")
            # # plt.close()
                        
            ## Compute metrics                      
            psnr_val, ssim_val, mse_val = compute_metrics_2D(img_full, img_recon)    
            
            df.loc[k] = [acc_factor, patient_name, mat_file, psnr_val, ssim_val, mse_val]
            k+=1
            
            
            #create or over-write (update) the CSV file:
            try:
                df.to_csv(os.path.join(os.getcwd(),f"metrics_{name_of_reconstrucedImages_folder}_{name_of_partition}.csv"))
            except PermissionError: #in case you are taking a look at the current CSV while it tries to write it again, wait 6 seconds
                time.sleep(6)