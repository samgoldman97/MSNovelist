""" Fingerprint smiles """
import os
import sys
import argparse
from pathlib import Path
import pickle
import numpy as np
import subprocess

sys.path.append("/msnovelist")

parser = argparse.ArgumentParser()
parser.add_argument("--smi-list", type=str, help="Name of smi file",
                    default="all_smis.txt"
                    )
parser.add_argument("--mounted-dir", type=str, help="Name of mounted volume",
                    default="/fp_data"
                    )
parser.add_argument("--uid", type=str, help="Name of owner",
                    default="1002"
                    )
parser.add_argument("--fp-map", type=str,
                    help="Name of fp mapping file",
                    default=None)
args = parser.parse_args()

# Remove arguments to avoid conflict with secondary arguments
while len(sys.argv) > 1:
    del sys.argv[1]

import fp_management.fingerprinting as fpr
import smiles_config as sc

smi_list = args.smi_list
user = args.uid
mounted_dir = Path(args.mounted_dir)
fp_map = args.fp_map

if smi_list is None:
    smis = ["CCC"]
else:
    in_file = mounted_dir / smi_list
    if not in_file.exists():
        raise ValueError()

    smis = [i.strip() for i in open(in_file, "rb").readlines()]

path = sc.config['fingerprinter_path']
threads = 30
fpr.Fingerprinter.init_instance(path,
                                threads,
                                capture=True)
fingerprinter = fpr.Fingerprinter.get_instance()

out_fps = fingerprinter.process(smis, calc_fingerprint=True,
                                return_b64=True)
full_output = [(fpr.get_fp(i['fingerprint']), i['smiles_canonical'])
               for i in out_fps]
full_output = [(i, j) for i, j in full_output if i is not None]
full_output, full_smis = zip(*full_output)
full_output = np.vstack(full_output)

if fp_map is not None:
    fp_map = pickle.load(open(fp_map, "rb"))
    num_to_set = len(fp_map)
    col_ind = np.zeros(num_to_set)
    for k, v in fp_map.items():
        col_ind[k] = v
    col_ind = col_ind.astype(int)
    full_output = full_output[:, col_ind]

# Output to file
output_dir = mounted_dir / "fp_out"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "output.p"
with open(output_file, "wb") as fp:
    pickle.dump((full_output, full_smis), fp)

# Mod permissions of the file
mod_cmd = f'chown -R {user} {output_dir}'
subprocess.call(mod_cmd, shell=True)
