
import os
import sys

SMI_DIR = os.environ['SMI_DIR']
SMI_OUTPUT = os.environ['SMI_OUTPUT']

print (f'SMI_DIR = {SMI_DIR}')
print (f'SMI_OUTPUT = {SMI_OUTPUT}')

#is_training = True
is_training = False

input_params = []

if is_training:

    input_params += [ ['1ES0647'               ,102.694 ,25.050 , 'OFF'] ]
    input_params += [ ['1ES1011'               ,153.767 ,49.434 , 'OFF'] ]
    input_params += [ ['1ES0414'               ,64.220  ,1.089  , 'OFF'] ]
    input_params += [ ['1ES0502'               ,76.983  ,67.623 , 'OFF'] ]
    input_params += [ ['1ES0229'               ,38.222  ,20.273 , 'OFF'] ]
    input_params += [ ['M82'                   ,148.970 ,69.679 , 'OFF'] ]
    input_params += [ ['3C264'                 ,176.271 ,19.606 , 'OFF'] ]
    input_params += [ ['BLLac'                 ,330.680 ,42.277 , 'OFF'] ]
    input_params += [ ['Draco'                 ,260.059 ,57.921 , 'OFF'] ]
    input_params += [ ['OJ287'                 ,133.705 ,20.100 , 'OFF'] ]
    input_params += [ ['H1426'                 ,217.136  ,42.673, 'OFF' ] ]
    input_params += [ ['NGC1275'               ,49.950  ,41.512 , 'OFF'] ]
    input_params += [ ['Segue1'                ,151.767 ,16.082 , 'OFF'] ]
    input_params += [ ['3C273'                 ,187.277 ,2.05   , 'OFF'] ]
    input_params += [ ['PG1553'                ,238.936 ,11.195 , 'OFF'] ]
    input_params += [ ['PKS1424'               ,216.750 ,23.783 , 'OFF'] ]
    input_params += [ ['RGB_J0710_p591'        ,107.61  ,59.15  , 'OFF'] ]
    input_params += [ ['UrsaMinor'             ,227.285 ,67.222 , 'OFF'] ]
    input_params += [ ['UrsaMajorII'           ,132.875 ,63.13  , 'OFF'] ]
    input_params += [ ['1ES1959_p650'          ,300.00 ,65.15   , 'OFF'] ]
    input_params += [ ['CrabNebula_elev_80_90' ,83.633  ,22.014 , 'OFF'] ]
    input_params += [ ['CrabNebula_elev_70_80' ,83.633  ,22.014 , 'OFF'] ]
    input_params += [ ['CrabNebula_elev_60_70' ,83.633  ,22.014 , 'OFF'] ]
    input_params += [ ['CrabNebula_elev_50_60' ,83.633  ,22.014 , 'OFF'] ]
    input_params += [ ['CrabNebula_elev_40_50' ,83.633  ,22.014 , 'OFF'] ]
    input_params += [ ['CrabNebula_elev_30_40' ,83.633  ,22.014 , 'OFF'] ]
    input_params += [ ['CrabNebula_1p0wobble' ,83.633  ,22.014 , 'OFF'] ]
    input_params += [ ['CrabNebula_1p5wobble' ,83.633  ,22.014 , 'OFF'] ]
    
else:

    #input_params += [ ['1ES0647'               ,102.694 ,25.050 , 'ON'] ]
    #input_params += [ ['1ES1011'               ,153.767 ,49.434 , 'ON'] ]
    #input_params += [ ['1ES0414'               ,64.220  ,1.089  , 'ON'] ]
    #input_params += [ ['1ES0502'               ,76.983  ,67.623 , 'ON'] ]
    #input_params += [ ['1ES0229'               ,38.222  ,20.273 , 'ON'] ]
    #input_params += [ ['M82'                   ,148.970 ,69.679 , 'ON'] ]
    #input_params += [ ['3C264'                 ,176.271 ,19.606 , 'ON'] ]
    #input_params += [ ['BLLac'                 ,330.680 ,42.277 , 'ON'] ]
    #input_params += [ ['Draco'                 ,260.059 ,57.921 , 'ON'] ]
    #input_params += [ ['OJ287'                 ,133.705 ,20.100 , 'ON'] ]
    #input_params += [ ['H1426'                 ,217.136  ,42.673, 'ON' ] ]
    #input_params += [ ['NGC1275'               ,49.950  ,41.512 , 'ON'] ]
    #input_params += [ ['Segue1'                ,151.767 ,16.082 , 'ON'] ]
    #input_params += [ ['3C273'                 ,187.277 ,2.05   , 'ON'] ]
    #input_params += [ ['PG1553'                ,238.936 ,11.195 , 'ON'] ]
    #input_params += [ ['PKS1424'               ,216.750 ,23.783 , 'ON'] ]
    #input_params += [ ['RGB_J0710_p591'        ,107.61  ,59.15  , 'ON'] ]
    #input_params += [ ['UrsaMinor'             ,227.285 ,67.222 , 'ON'] ]
    #input_params += [ ['UrsaMajorII'           ,132.875 ,63.13  , 'ON'] ]
    #input_params += [ ['1ES1959_p650'          ,300.00 ,65.15   , 'ON'] ]
    input_params += [ ['CrabNebula_elev_80_90' ,83.633  ,22.014 , 'ON'] ]
    input_params += [ ['CrabNebula_elev_70_80' ,83.633  ,22.014 , 'ON'] ]
    input_params += [ ['CrabNebula_elev_60_70' ,83.633  ,22.014 , 'ON'] ]
    input_params += [ ['CrabNebula_elev_50_60' ,83.633  ,22.014 , 'ON'] ]
    input_params += [ ['CrabNebula_elev_40_50' ,83.633  ,22.014 , 'ON'] ]
    input_params += [ ['CrabNebula_elev_30_40' ,83.633  ,22.014 , 'ON'] ]
    input_params += [ ['CrabNebula_1p0wobble' ,83.633  ,22.014 , 'ON'] ]
    input_params += [ ['CrabNebula_1p5wobble' ,83.633  ,22.014 , 'ON'] ]
    
    input_params += [ ['SNR_G189_p03'          ,94.213  ,22.503, 'ON' ] ] # ic 443
    input_params += [ ['PSR_J1907_p0602'       ,286.975 ,6.337 , 'ON' ] ]
    input_params += [ ['PSR_J2021_p4026'       ,305.37  ,40.45 , 'ON' ] ] # gamma cygni
    input_params += [ ['PSR_J2021_p3651'       ,305.27  ,36.85 , 'ON' ] ] # Dragonfly
    input_params += [ ['PSR_J2032_p4127', 308.05  , 41.46 , 'ON' ] ]
    input_params += [ ['PSR_J2032_p4127_baseline', 308.05  , 41.46 , 'ON'  ] ]
    input_params += [ ['PSR_J1856_p0245', 284.21  , 2.76 , 'ON' ] ]
    input_params += [ ['SS433'       ,288.404, 4.930 , 'ON' ] ]
    input_params += [ ['PSR_J1928_p1746', 292.15, 17.78 , 'ON' ] ]
    input_params += [ ['LHAASO_J0622_p3754', 95.50  , 37.90 , 'ON' ] ]
    input_params += [ ['Cas_A', 350.8075  , 58.8072 , 'ON' ] ]
    input_params += [ ['PSR_J2229_p6114', 337.27, 61.23 , 'ON' ] ] # Boomerang
    input_params += [ ['Geminga'               ,98.476  ,17.770 , 'ON' ] ]
    input_params += [ ['SNR_G150_p4', 66.785, 55.458 , 'ON' ] ] # Jamie's SNR
    input_params += [ ['CTA1', 1.608, 72.983 , 'ON' ] ]
    input_params += [ ['Tycho', 6.28  , 64.17 , 'ON'  ] ]
    input_params += [ ['2HWC_J1953_p294', 298.26 , 29.48 , 'ON' ] ]
    input_params += [ ['PSR_J2238_p5903', 339.50 , 59.05 , 'ON' ] ]
    
    #input_params += [ ['2FHL_J0431_p5553e', 67.81 , 55.89 , 'ON' ] ]
    #input_params += [ ['CTB109', 345.28 , 58.88 , 'ON' ] ]
    #input_params += [ ['PSR_B1937_p21', 295.45 , 21.44 , 'ON' ] ]
    #input_params += [ ['RX_J0648_p1516', 102.20 , 15.27 , 'ON' ] ]
    #input_params += [ ['LS_V_p4417', 70.25 , 44.53 , 'ON' ] ]
    
    #input_params += [ ['PSR_J1747_m2809', 266.825  , -28.15 , 'ON' ] ] # Sgr A*

for s in range(0,len(input_params)):
    source = input_params[s][0]
    file = open("run/save_mtx_%s.sh"%(source),"w") 
    file.write('cd %s\n'%(SMI_DIR))
    file.write(f'python3 save_big_matrices.py "{source}" "V6"\n')
    file.write(f'python3 save_big_matrices.py "{source}" "V5"\n')
    file.write(f'python3 save_big_matrices.py "{source}" "V4"\n')
    file.close() 

qfile = open("run/sub_condor_save_mtx.sh","w") 
for s in range(0,len(input_params)):
    source = input_params[s][0]
    qfile.write('universe = vanilla \n')
    qfile.write('getenv = true \n')
    qfile.write('executable = /bin/bash \n')
    qfile.write('arguments = save_mtx_%s.sh\n'%(source))
    qfile.write('request_cpus = 1 \n')
    qfile.write('request_memory = 1024M \n')
    qfile.write('request_disk = 1024M \n')
    qfile.write('output = condor_save_mtx_%s.out\n'%(source))
    qfile.write('error = condor_save_mtx_%s.err\n'%(source))
    qfile.write('log = condor_save_mtx_%s.log\n'%(source))
    qfile.write('queue\n')
qfile.close() 

for s in range(0,len(input_params)):
    source = input_params[s][0]
    file = open("run/eigenvtr_%s.sh"%(source),"w") 
    file.write('cd %s\n'%(SMI_DIR))
    file.write(f'python3 build_eigenvectors.py "{source}" "V6"\n')
    file.write(f'python3 build_eigenvectors.py "{source}" "V5"\n')
    file.write(f'python3 build_eigenvectors.py "{source}" "V4"\n')
    file.close() 

qfile = open("run/sub_condor_eigenvtr.sh","w") 
for s in range(0,len(input_params)):
    source = input_params[s][0]
    qfile.write('universe = vanilla \n')
    qfile.write('getenv = true \n')
    qfile.write('executable = /bin/bash \n')
    qfile.write('arguments = eigenvtr_%s.sh\n'%(source))
    qfile.write('request_cpus = 1 \n')
    qfile.write('request_memory = 1024M \n')
    qfile.write('request_disk = 1024M \n')
    qfile.write('output = condor_eigenvtr_%s.out\n'%(source))
    qfile.write('error = condor_eigenvtr_%s.err\n'%(source))
    qfile.write('log = condor_eigenvtr_%s.log\n'%(source))
    qfile.write('queue\n')
qfile.close() 

for s in range(0,len(input_params)):
    source = input_params[s][0]
    src_ra = input_params[s][1]
    src_dec = input_params[s][2]
    onoff = input_params[s][3]
    file = open("run/skymap_%s_%s.sh"%(source,onoff),"w") 
    file.write('cd %s\n'%(SMI_DIR))
    file.write(f'python3 save_skymaps.py "{source}" {src_ra} {src_dec} "{onoff}" "V6"\n')
    file.write(f'python3 save_skymaps.py "{source}" {src_ra} {src_dec} "{onoff}" "V5"\n')
    file.write(f'python3 save_skymaps.py "{source}" {src_ra} {src_dec} "{onoff}" "V4"\n')
    file.close() 

qfile = open("run/sub_condor_skymap.sh","w") 
for s in range(0,len(input_params)):
    source = input_params[s][0]
    onoff = input_params[s][3]
    qfile.write('universe = vanilla \n')
    qfile.write('getenv = true \n')
    qfile.write('executable = /bin/bash \n')
    qfile.write('arguments = skymap_%s_%s.sh\n'%(source,onoff))
    qfile.write('request_cpus = 1 \n')
    qfile.write('request_memory = 1024M \n')
    qfile.write('request_disk = 1024M \n')
    qfile.write('output = condor_skymap_%s_%s.out\n'%(source,onoff))
    qfile.write('error = condor_skymap_%s_%s.err\n'%(source,onoff))
    qfile.write('log = condor_skymap_%s_%s.log\n'%(source,onoff))
    qfile.write('queue\n')
qfile.close() 

for s in range(0,len(input_params)):
    source = input_params[s][0]
    src_ra = input_params[s][1]
    src_dec = input_params[s][2]
    onoff = input_params[s][3]
    file = open("run/plot_%s_%s.sh"%(source,onoff),"w") 
    file.write('cd %s\n'%(SMI_DIR))
    #file.write('sh clean_plots.sh\n')
    file.write(f'python3 plot_analysis_result.py "{source}" {src_ra} {src_dec} "{onoff}" \n')
    file.close() 

