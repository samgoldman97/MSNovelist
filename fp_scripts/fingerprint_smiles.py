""" Fingerprint smiles """

import os
import sys

sys.path.append("/msnovelist")
sys.path.append("/msnovelist/fingerprint-wrapper/src/")
sys.path.append("/msnovelist/fingerprint-wrapper/")

import smiles_config as sc
import fp_management.fingerprinting as fpr

sample_smiles = "CCC"

path = sc.config['fingerprinter_path']
threads = sc.config['fingerprinter_threads']
fpr.Fingerprinter.init_instance(path,
                                threads,
                                capture=True)
fingerprinter = fpr.Fingerprinter.get_instance()


smi_list = [sample_smiles]
out_fps = fingerprinter.process(smi_list, calc_fingerprint=True,
                                return_b64=True)
fp = out_fps[0]['fingerprint']

# Size 8925
out = fpr.get_fp(fp)
print(out.shape)
