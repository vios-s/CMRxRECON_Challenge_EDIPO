import os
from pathlib import Path

import scipy.io as scio
def save_reconstructions(recons, save_dir):
    for fname, recon in recons.items():
        file_parts = fname.split("/")
        out_dir = Path(os.path.join(save_dir, 'Submission/SingleCoil/Cine/TestSet', file_parts[-3], file_parts[-2]))
        out_dir.mkdir(exist_ok=True, parents=True)
        path = (out_dir / file_parts[-1]).resolve()
        save_dict = {'img4ranking': recon.transpose(1, 2, 0, 3)}
        scio.savemat(file_name=str(path),
                     mdict=save_dict)

# import numpy as np
# def save_reconstructions(recons, save_dir):
#
#     print("SAVE RECONSTRUCTIONS...")
#     for fname, recon in recons.items():
#
#         file_parts = fname.split("/")
# <<<<<<< HEAD
#         out_dir = Path(os.path.join(save_dir, 'Submission/SingleCoil/Cine/TestSet',file_parts[-3], file_parts[-2]))
# =======
#
#         out_dir = Path(os.path.join(save_dir, file_parts[-3], file_parts[-2]))
# >>>>>>> main
#         out_dir.mkdir(exist_ok=True, parents=True)
#
#         path = (out_dir / file_parts[-1]).resolve()
#         #e.g., "Y:\raid\home\gianlucacarloni\CMRxRecon\output_5c_npy\AccFactor04\P111\cine_sax.mat"
#
#         recon_to_be_saved = recon.transpose(1,2,0,3)
#         np.save(path,recon_to_be_saved)
#         #e.g., "Y:\raid\home\gianlucacarloni\CMRxRecon\output_5c_npy\AccFactor04\P111\cine_sax.mat.npy"
