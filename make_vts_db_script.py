
import os
import sys

SMI_DIR = os.environ['SMI_DIR']

input_params = []

input_params += [ ['1ES0647'               ,102.694 ,25.050 ,30 ,90] ]
input_params += [ ['1ES1011'               ,153.767 ,49.434 ,30 ,90] ]
input_params += [ ['1ES0414'               ,64.220  ,1.089  ,30 ,90] ]
input_params += [ ['1ES0502'               ,76.983  ,67.623 ,30 ,90] ]
input_params += [ ['1ES0229'               ,38.222  ,20.273 ,30 ,90] ]
input_params += [ ['M82'                   ,148.970 ,69.679 ,30 ,90] ]
input_params += [ ['3C264'                 ,176.271 ,19.606 ,30 ,90] ]
input_params += [ ['BLLac'                 ,330.680 ,42.277 ,30 ,90] ]
input_params += [ ['Draco'                 ,260.059 ,57.921 ,30 ,90] ]
input_params += [ ['OJ287'                 ,133.705 ,20.100 ,30 ,90] ]
input_params += [ ['H1426'                 ,217.136  ,42.673 ,30 ,90] ]
input_params += [ ['NGC1275'               ,49.950  ,41.512 ,30 ,90] ]
input_params += [ ['Segue1'                ,151.767 ,16.082 ,30 ,90] ]
input_params += [ ['3C273'                 ,187.277 ,2.05   ,30 ,90] ]
input_params += [ ['PG1553'                ,238.936 ,11.195 ,30 ,90] ]
input_params += [ ['PKS1424'               ,216.750 ,23.783 ,30 ,90] ]
input_params += [ ['RGB_J0710_p591'        ,107.61  ,59.15  ,30 ,90] ]
input_params += [ ['UrsaMinor'             ,227.285 ,67.222 ,30 ,90] ]
input_params += [ ['UrsaMajorII'           ,132.875 ,63.13  ,30 ,90] ]
input_params += [ ['CrabNebula_elev_80_90' ,83.633  ,22.014 ,80 ,90] ]
input_params += [ ['CrabNebula_elev_70_80' ,83.633  ,22.014 ,70 ,80] ]
input_params += [ ['CrabNebula_elev_60_70' ,83.633  ,22.014 ,60 ,70] ]
input_params += [ ['CrabNebula_elev_50_60' ,83.633  ,22.014 ,50 ,60] ]
input_params += [ ['CrabNebula_elev_40_50' ,83.633  ,22.014 ,40 ,50] ]
input_params += [ ['CrabNebula_elev_30_40' ,83.633  ,22.014 ,30 ,40] ]
input_params += [ ['CrabNebula_1p0wobble' ,83.633  ,22.014 ,30 ,90] ]
input_params += [ ['CrabNebula_1p5wobble' ,83.633  ,22.014 ,30 ,90] ]
input_params += [ ['1ES1959_p650'          ,300.00 ,65.15 ,30 ,90] ]

input_params += [ ['SNR_G189_p03'          ,94.213  ,22.503 ,30 ,90] ] # ic 443
input_params += [ ['PSR_J1907_p0602'       ,286.975 ,6.337  ,30 ,90] ]
input_params += [ ['PSR_J2021_p4026'       ,305.37  ,40.45  ,30 ,90] ] # gamma cygni
input_params += [ ['PSR_J2021_p3651'       ,305.27  ,36.85  ,30 ,90] ] # Dragonfly
input_params += [ ['PSR_J2032_p4127', 308.05  , 41.46  ,30 ,90] ]
input_params += [ ['PSR_J2032_p4127_baseline', 308.05  , 41.46  ,30 ,90] ]
input_params += [ ['PSR_J1856_p0245', 284.21  , 2.76  ,30 ,90] ]
input_params += [ ['SS433'       ,288.404, 4.930  ,30 ,90] ]
input_params += [ ['PSR_J1928_p1746', 292.15, 17.78, 30 ,90] ]
input_params += [ ['LHAASO_J0622_p3754', 95.50  , 37.90  ,30 ,90] ]
input_params += [ ['Cas_A', 350.8075  , 58.8072  ,30 ,90] ]
input_params += [ ['PSR_J2229_p6114', 337.27, 61.23, 30 ,90] ] # Boomerang

input_params += [ ['Geminga'               ,98.476  ,17.770 ,30 ,90] ]
input_params += [ ['CTA1', 1.608, 72.983  ,30 ,90] ]

#input_params += [ ['Tycho', 6.28  , 64.17  ,30 ,90] ]
#input_params += [ ['SNR_G150_p4', 66.785, 55.458,30 ,90] ] # Jamie's SNR
#input_params += [ ['2HWC_J1953_p294', 298.26 , 29.48 ,30 ,90] ]
#input_params += [ ['2FHL_J0431_p5553e', 67.81 , 55.89 ,30 ,90] ]
#input_params += [ ['PSR_B1937_p21', 295.45 , 21.44 ,30 ,90] ]
#input_params += [ ['RX_J0648_p1516', 102.20 , 15.27 ,30 ,90] ]
#input_params += [ ['LS_V_p4417', 70.25 , 44.53 ,30 ,90] ]

#input_params += [ ['PSR_J1747_m2809', 266.825  , -28.15  ,30 ,90] ] # Sgr A*



#input_params += [ ['Galactic_Plane', 0., 0.  ,30 ,90] ]
#input_params += [ ['Extragalactic', 0., 0.  ,30 ,90] ]
#input_params += [ ['LHAASO_Catalog', 0., 0.  ,30 ,90] ]

#input_params += [ ['AUX_files', 0., 0.  ,30 ,90] ]

job_counts = 0
for s in range(0,len(input_params)):
    job_counts += 1
    file = open("run/vts_db_%s.sh"%(input_params[s][0]),"w") 
    file.write('cd %s\n'%(SMI_DIR))
    if input_params[s][0]=='AUX_files':
        file.write('python3 veritas_db_query.py "%s" %s %s %s %s "V4V5V6"\n'%(input_params[s][0],input_params[s][1],input_params[s][2],input_params[s][3],input_params[s][4]))
    else:
        file.write('python3 veritas_db_query.py "%s" %s %s %s %s "V6"\n'%(input_params[s][0],input_params[s][1],input_params[s][2],input_params[s][3],input_params[s][4]))
        file.write('python3 veritas_db_query.py "%s" %s %s %s %s "V5"\n'%(input_params[s][0],input_params[s][1],input_params[s][2],input_params[s][3],input_params[s][4]))
        file.write('python3 veritas_db_query.py "%s" %s %s %s %s "V4"\n'%(input_params[s][0],input_params[s][1],input_params[s][2],input_params[s][3],input_params[s][4]))
    file.close() 

job_counts = 0
qfile = open("run/condor_vts_db.sh","w") 
for s in range(0,len(input_params)):
    job_counts += 1
    qfile.write('universe = vanilla \n')
    qfile.write('getenv = true \n')
    qfile.write('executable = /bin/bash \n')
    qfile.write('arguments = vts_db_%s.sh\n'%(input_params[s][0]))
    qfile.write('request_cpus = 1 \n')
    qfile.write('request_memory = 1024M \n')
    qfile.write('request_disk = 1024M \n')
    qfile.write('output = condor_vts_db_%s.out\n'%(input_params[s][0]))
    qfile.write('error = condor_vts_db_%s.err\n'%(input_params[s][0]))
    qfile.write('log = condor_vts_db_%s.log\n'%(input_params[s][0]))
    qfile.write('queue\n')
qfile.close() 

