
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

big_off_matrix, big_mask_matrix, big_off_matrix_fullspec, big_mask_matrix_fullspec = build_big_camera_matrix(source_name,src_ra,src_dec,smi_input,off_runlist,max_runs=1e10,is_bkgd=True)

#on_file = f'{smi_runlist}/RunList_{source_name}_{input_epoch}.txt'
#off_file = f'{smi_runlist}/PairList_{source_name}_{input_epoch}.txt'
#mimic_file = f'{smi_runlist}/ImposterList_{source_name}_{input_epoch}.txt'
#on_runlist, off_runlist, mimic_runlist = ReadRunListFromFile(smi_input,on_file,off_file,mimic_file)
#big_off_matrix = []
#big_off_matrix_fullspec = []
#for batch in range(0,len(off_runlist)):
#    print (f"batch = {batch}/{len(off_runlist)}")
#    run_batch = off_runlist[batch]
#    off_matrix, mask_matrix, off_matrix_fullspec, mask_matrix_fullspec = build_big_camera_matrix(source_name,src_ra,src_dec,smi_input,run_batch,max_runs=1e10,is_bkgd=True)
#    if len(off_matrix)==0: continue
#    sum_matrix = np.zeros_like(off_matrix[0])
#    for run in range(0,len(off_matrix)):
#        for logE in range(0,logE_nbins):
#            for pix in range(0,len(sum_matrix[logE])):
#                sum_matrix[logE][pix] += off_matrix[run][logE][pix]
#    if len(off_matrix_fullspec)==0: continue
#    sum_matrix_fullspec = np.zeros_like(off_matrix_fullspec[0])
#    for run in range(0,len(off_matrix_fullspec)):
#        for pix in range(0,len(sum_matrix_fullspec)):
#            sum_matrix_fullspec[pix] += off_matrix_fullspec[run][pix]
#    big_off_matrix += [sum_matrix]
#    big_off_matrix_fullspec += [sum_matrix_fullspec]


#input_params = []
#input_params += [ ['1ES0647'               ,102.694 ,25.050 , 'OFF'] ]
#input_params += [ ['1ES1011'               ,153.767 ,49.434 , 'OFF'] ]
#input_params += [ ['1ES0414'               ,64.220  ,1.089  , 'OFF'] ]
#input_params += [ ['1ES0502'               ,76.983  ,67.623 , 'OFF'] ]
#input_params += [ ['1ES0229'               ,38.222  ,20.273 , 'OFF'] ]
#input_params += [ ['M82'                   ,148.970 ,69.679 , 'OFF'] ]
#input_params += [ ['3C264'                 ,176.271 ,19.606 , 'OFF'] ]
#input_params += [ ['BLLac'                 ,330.680 ,42.277 , 'OFF'] ]
#input_params += [ ['Draco'                 ,260.059 ,57.921 , 'OFF'] ]
#input_params += [ ['OJ287'                 ,133.705 ,20.100 , 'OFF'] ]
#input_params += [ ['H1426'                 ,217.136  ,42.673, 'OFF' ] ]
#input_params += [ ['NGC1275'               ,49.950  ,41.512 , 'OFF'] ]
#input_params += [ ['Segue1'                ,151.767 ,16.082 , 'OFF'] ]
#input_params += [ ['3C273'                 ,187.277 ,2.05   , 'OFF'] ]
#input_params += [ ['PG1553'                ,238.936 ,11.195 , 'OFF'] ]
#input_params += [ ['PKS1424'               ,216.750 ,23.783 , 'OFF'] ]
#input_params += [ ['RGB_J0710_p591'        ,107.61  ,59.15  , 'OFF'] ]
#input_params += [ ['UrsaMinor'             ,227.285 ,67.222 , 'OFF'] ]
#input_params += [ ['UrsaMajorII'           ,132.875 ,63.13  , 'OFF'] ]
#input_params += [ ['1ES1959_p650'          ,300.00 ,65.15   , 'OFF'] ]
#
#all_runlist = []
#for s in range(0,len(input_params)):
#    source = input_params[s][0]
#    src_ra = input_params[s][1]
#    src_dec = input_params[s][2]
#    onoff = input_params[s][3]
#
#    if source_name==source: continue
#
#    on_file = f'{smi_runlist}/RunList_{source}_{input_epoch}.txt'
#    off_file = f'{smi_runlist}/PairList_{source}_{input_epoch}.txt'
#    mimic_file = f'{smi_runlist}/ImposterList_{source}_{input_epoch}.txt'
#    on_runlist, off_runlist, mimic_runlist = ReadRunListFromFile(smi_input,on_file,off_file,mimic_file)
#
#    all_runlist += on_runlist
#
#big_off_matrix, big_mask_matrix, big_off_matrix_fullspec, big_mask_matrix_fullspec = build_big_camera_matrix(source_name,src_ra,src_dec,smi_input,all_runlist,max_runs=1e10,is_bkgd=True)





output_filename = f'{smi_output}/big_off_matrix_{source_name}_{onoff}_{input_epoch}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump([big_off_matrix,big_off_matrix_fullspec], file)


print ('Big matrices saved.')
