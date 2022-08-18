""" Fingerprint smiles """
import sys
import argparse
from pathlib import Path
import pickle
import numpy as np
import h5py
import subprocess
from rdkit import Chem
import multiprocessing as mp
from tqdm import tqdm

from functools import partial
from itertools import islice

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
parser.add_argument("--out-prefix", type=str, help="Output prefix",
                    default="precomputed"
                    )
parser.add_argument("--fp-map", type=str,
                    help="Name of fp mapping file",
                    default=None)
parser.add_argument("--workers", type=int,
                    default=30,
                    help="Num workers",)
args = parser.parse_args()

def batch_func(list_inputs, function):
    outputs = []
    for i in list_inputs:
        outputs.append(function(i))
    return outputs

def chunked_parallel(input_list, function, chunks=100,
                     max_cpu=16):
    """chunked_parallel.

    Args:
        input_list : list of objects to apply function
        function : Callable with 1 input and returning a single value
        chunks: number of hcunks
        max_cpu: Max num cpus
    """
    # Adding it here fixes somessetting disrupted elsewhere
    list_len = len(input_list)
    num_chunks = min(list_len, chunks)
    step_size = len(input_list) // num_chunks

    chunked_list = [input_list[i:i+step_size]
                    for i in range(0, len(input_list), step_size)]

    partial_func = partial(batch_func, function=function)

    list_outputs = simple_parallel(chunked_list,
                                   partial_func,
                                   max_cpu=max_cpu)
    # Unroll
    full_output = [j for i in list_outputs for j in i]
    return full_output

def simple_parallel(input_list, function, max_cpu=16): 
    """ Simple parallelization.

    Use map async and retries in case we get odd stalling behavior.

    input_list: Input list to op on
    function: Fn to apply
    max_cpu: Num cpus

    """
    cpus = min(mp.cpu_count(), max_cpu)
    pool = mp.Pool(processes=cpus)
    async_results = [pool.apply_async(function, args=(i, ))
                     for i in input_list]
    pool.close()
    list_outputs = []
    for async_result in tqdm(async_results, total=len(input_list)):
        result = async_result.get()
        list_outputs.append(result)
    pool.join()
    return list_outputs

def get_inchikey(smi):
    try:
        return Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
    except:
        return None

# Remove arguments to avoid conflict with secondary arguments
while len(sys.argv) > 1:
    del sys.argv[1]

import fp_management.fingerprinting as fpr
import smiles_config as sc

smi_list = args.smi_list
user = args.uid
mounted_dir = Path(args.mounted_dir)
fp_map = args.fp_map
workers = args.workers
out_prefix = args.out_prefix

if smi_list is None:
    smis = ["CCC"]
else:
    in_file = mounted_dir / smi_list
    if not in_file.exists():
        raise ValueError()

    smis = [i.strip() for i in open(in_file, "r").readlines()]

path = sc.config['fingerprinter_path']


def batches(it, chunk_size):
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])

def get_fps(smiles):
    import fp_management.fingerprinting as fpr

    fpr.Fingerprinter.init_instance(path,
                                    workers,
                                    capture=True)
    fingerprinter = fpr.Fingerprinter.get_instance()
    out_fps = fingerprinter.process(smiles, calc_fingerprint=True,
                                    return_b64=True)
    full_output = [(fpr.get_fp(i['fingerprint']), i['smiles_canonical'])
                   for i in out_fps]
    return full_output

smis = smis
batched_smis = list(batches(smis, 10000))
import time
start = time.time()
out_smis = simple_parallel(batched_smis, get_fps, max_cpu=workers)
#out_smis = [get_fps(j) for j in batched_smis]
end = time.time()
print(f"TIME: {end - start}")
full_output = [j for i in out_smis for j in i]
full_output, full_smis = zip(*full_output)
#raise ValueError("Debugging, done with code")

# Create inchikeys
inchikeys = chunked_parallel(full_smis, get_inchikey, chunks=200,
                             max_cpu=workers)

full_output = [(i.squeeze(), j) for i, j in zip(full_output, inchikeys)
               if j is not None and i is not None]
full_output, full_inchikeys = zip(*full_output)
full_output = np.vstack(full_output)

# Subset down to appropriate fp size
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
output_index_file = output_dir / f"{out_prefix}_index.p"
output_hdf_file = output_dir / f"{out_prefix}.hdf5"

with open(output_index_file, "wb") as fp:
    key_to_ind = dict(zip(full_inchikeys, np.arange(len(full_inchikeys))))
    pickle.dump(key_to_ind, fp)

print("Dumping to h5py")
h = h5py.File(output_hdf_file, "w")
dset = h.create_dataset("features", data=full_output, 
                        dtype=np.uint8)

# Mod permissions of the file
mod_cmd = f'chown -R {user} {output_dir}'
subprocess.call(mod_cmd, shell=True)
