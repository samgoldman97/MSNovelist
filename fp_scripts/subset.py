""" Map fingerprints shared by Kai to fingerprints output by program"""
import argparse
import h5py

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from rdkit import Chem
from collections import Counter


debug = False

# Step 1: Get map from smi to fingerprints from Kai
labels_file = "../MassSpec/data/paired_spectra/csi2022/labels.tsv"
features_hdf = "../MassSpec/fingerprints/precomputed_fp/cache_csi_csi2022.hdf5"
features_p = "../MassSpec/fingerprints/precomputed_fp/cache_csi_csi2022_index.p"

df = pd.read_csv(labels_file, sep="\t")
smiles_to_inchikey = dict(df[["smiles", "inchikey"]].values)

inchikey_list = list(set(list(df['inchikey'].values)))
inchikey_set = set(inchikey_list)

# 2. Load featurizations
print("Loading old features")
features_pickle_file = pickle.load(open(features_p, "rb"))
features_hdf = h5py.File(features_hdf, "r")['features'][:]
old_inchikey_to_fps = {k: features_hdf[v] for k,v in features_pickle_file.items()}


# Step 2: Extract smi to fingerprints from this program
print("Loading new features")
large_fingerprints = "fp_data/fp_out/all_smis.p"
new_fingerprints, new_smis = pickle.load(open(large_fingerprints, "rb"))

if debug:
    new_smis = new_smis[:500] # Debug

inchikeys = [Chem.MolToInchiKey(Chem.MolFromSmiles(j)) for j in new_smis]
new_inchikey_to_fps = dict(zip(inchikeys, new_fingerprints))

print(len(old_inchikey_to_fps))
print(len(new_inchikey_to_fps))

print("Creating row aligned fp matrices")
fps_old, fps_new = [], []
for k in new_inchikey_to_fps:
    if k not in old_inchikey_to_fps:
        continue
    fps_old.append(old_inchikey_to_fps.get(k))
    fps_new.append(new_inchikey_to_fps.get(k))

fps_old, fps_new = np.vstack(fps_old), np.vstack(fps_new)

print("Shape mat 1:", fps_old.shape)
print("Shape mat 2:", fps_new.shape)

# Step 4: Convert columns into bitstrings
print("Converting to bitstrings")
old_bitstrings = [fps_old[:, k].tostring() for k in range(fps_old.shape[1])]
old_bitstrings = np.array(old_bitstrings)

new_bitstrings = [fps_new[:, k].tostring() for k in range(fps_new.shape[1])]
new_bitstrings = np.array(new_bitstrings)

# Map old to new
old_to_new  = {}
avail_to_match = np.ones(len(new_bitstrings))
for col_ind, col in enumerate(old_bitstrings):
    matching_inds = (col == new_bitstrings)
    matching_inds = np.logical_and(matching_inds, avail_to_match)

    map_ind = np.argwhere(matching_inds).flatten()[0]
    avail_to_match[map_ind] = 0
    old_to_new[col_ind] = map_ind

# Step 5: Output indices to file
output_file = "fp_map.p"
with open(output_file, "wb") as f: 
    pickle.dump(old_to_new, f)
