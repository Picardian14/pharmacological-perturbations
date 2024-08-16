from neuromaps.parcellate import Parcellater
from neuromaps import datasets
import numpy as np
from nilearn.datasets import fetch_atlas_aal
from neuromaps import nulls
import os
from scipy.io import savemat
PERMS=20
#aal = fetch_atlas_aal(version='SPM12')
aal_path = "/home/ivan.mindlin/Repos/neuromaps/AAL_90.nii.gz"
maps_path = "/home/ivan.mindlin/Repos/hansen_receptors/data/PET_nifti_images/"
receptor_fnames = [ '5HT2a_cimbi_hc29_beliveau',
 '5HT1a_cumi_hc8_beliveau',
 '5HT6_gsk_hc30_radhakrishnan',
 'D2_flb457_hc37_smith',     
 'D1_SCH23390_hc13_kaller',       
    'H3_cban_hc8_gallezot', 
 'MU_carfentanil_hc204_kantonen',
 'DAT_fpcit_hc174_dukart_spect']
#parc = Parcellater(aal['maps'],"MNI152").fit()
# Define a function that normalizes the intput vector receptor_org
def normalize_receptor(receptor_org):
   receptor_org = receptor_org[:90]
   receptor_parc = np.zeros_like(receptor_org)
   receptor_parc[:45] = receptor_org[::2]
   receptor_parc[45:] = receptor_org[1::2][::-1]
   receptor_orig = receptor_parc/(np.max(receptor_parc)-np.min(receptor_parc))
   receptor_norm = receptor_orig - np.max(receptor_orig) + 1    
   receptor_norm[receptor_norm<0]=0
   return receptor_norm

def normalize_null(vector, desired_mean):
    min_value = np.min(vector)
    shifted_vector = vector - min_value
    max_val = np.max(shifted_vector)
    scaled_vector = shifted_vector / max_val
    current_mean = np.mean(scaled_vector)
    scale_factor = desired_mean / current_mean
    normalized_vector = scaled_vector * scale_factor
    
    return normalized_vector

for fname in receptor_fnames:
   receptor_file = os.path.join("/network/lustre/iss02/cohen/data/Ivan/DMF_Gus/AAL/", fname+".csv")     
   receptor_org = np.genfromtxt(receptor_file, delimiter=',')   
   rotated = nulls.moran(receptor_org, atlas='MNI152', density='2mm',n_perm=PERMS, seed=1234, parcellation=aal_path)
   receptor_norm = normalize_receptor(receptor_org)
   rotated_norm = normalize_null(rotated, np.mean(receptor_norm))
   # Saving all 20 permutations. Use the fisrt one
   np.save(f"receptors/{fname}_MORAN_{PERMS}perm",rotated_norm)
   
