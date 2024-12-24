
import os, sys
import ROOT
import numpy as np
import pickle

import common_functions

logE_nbins = common_functions.logE_nbins
matrix_rank = common_functions.matrix_rank
ReadOffRunListFromFile = common_functions.ReadOffRunListFromFile
ReadRunListFromFile = common_functions.ReadRunListFromFile
build_big_camera_matrix = common_functions.build_big_camera_matrix

smi_runlist = os.environ.get("SMI_RUNLIST")
smi_input = os.environ.get("SMI_INPUT")
smi_output = os.environ.get("SMI_OUTPUT")
smi_dir = os.environ.get("SMI_DIR")
sky_tag = os.environ.get("SKY_TAG")
bin_tag = os.environ.get("BIN_TAG")

source_name = sys.argv[1]
src_ra = float(sys.argv[2])
src_dec = float(sys.argv[3])
onoff = sys.argv[4]
input_epoch = sys.argv[5] # 'V4', 'V5' or 'V6'
print (f'source_name = {source_name}, onoff = {onoff}, input_epoch = {input_epoch}')

off_runlist = []
if 'MIMIC' in onoff:
    off_runlist = ReadOffRunListFromFile(f'{smi_runlist}/ImposterList_{source_name}_{input_epoch}.txt',f'{smi_runlist}/ImposterPairList_{source_name}_{input_epoch}.txt',int(onoff.strip('MIMIC')))
else:
    off_runlist = ReadOffRunListFromFile(f'{smi_runlist}/RunList_{source_name}_{input_epoch}.txt',f'{smi_runlist}/PairList_{source_name}_{input_epoch}.txt',0)

for entry in range(0,len(off_runlist)):
    print (f"processing batch {entry}...")
    big_off_exposure, big_off_matrix_fullspec, big_mask_matrix_fullspec = build_big_camera_matrix(source_name,src_ra,src_dec,smi_input,off_runlist[entry],max_runs=1e10,is_bkgd=True)

    output_filename = f'{smi_output}/big_off_matrix_{source_name}_{onoff}_{input_epoch}_{bin_tag}_batch{entry}.pkl'
    with open(output_filename,"wb") as file:
        pickle.dump([big_off_exposure,big_off_matrix_fullspec], file)


print ('Big matrices saved.')
