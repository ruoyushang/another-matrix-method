
import os, sys
import ROOT
import numpy as np
import pickle
from scipy.optimize import least_squares, minimize

matrix_rank = 4

min_NImages = 3
max_Roff = 2.0
max_EmissionHeight_cut = 20.
min_EmissionHeight_cut = 6.
max_Rcore = 400.
min_Rcore = 0.
min_Energy_cut = 0.2
max_Energy_cut = 10.0
MSCW_cut = 0.5
MSCL_cut = 0.7
MVA_cut = 0.5

xoff_bins = 10
xoff_start = -2.
xoff_end = 2.
yoff_bins = 10
yoff_start = -2.
yoff_end = 2.
gcut_bins = 4
gcut_start = 0
gcut_end = gcut_bins
logE_bins = 5
logE_start = -1.+0.33
logE_end = 1.

smi_aux = os.environ.get("SMI_AUX")
smi_dir = os.environ.get("SMI_DIR")

def ReadRunListFromFile(input_file):

    runlist = []

    inputFile = open(input_file)
    for line in inputFile:
        runlist += [int(line)]

    return runlist

def ReadOffRunListFromFile(input_file):

    runlist = []

    inputFile = open(input_file)
    for line in inputFile:
        line_split = line.split()
        runlist += [int(line_split[1])]

    return runlist

def smooth_image(image_data,xaxis,yaxis,kernel_radius=0.07):

    image_smooth = np.zeros_like(image_data)

    bin_size = xaxis[1]-xaxis[0]

    kernel_pix_size = int(kernel_radius/bin_size)
    gaus_norm = 2.*np.pi*kernel_radius*kernel_radius
    image_kernel = np.zeros_like(image_data)
    central_bin_x = int(float(len(xaxis)-1)/2.)
    central_bin_y = int(float(len(yaxis)-1)/2.)
    for idx_x in range(0,len(xaxis)-1):
        for idx_y in range(0,len(yaxis)-1):
            pix_x = xaxis[idx_x] - xaxis[central_bin_x]
            pix_y = yaxis[idx_y] - yaxis[central_bin_y]
            distance = pow(pix_x*pix_x+pix_y*pix_y,0.5)
            pix_content = np.exp(-(distance*distance)/(2.*kernel_radius*kernel_radius))
            image_kernel[idx_y,idx_x] = pix_content
            #image_kernel[idx_y,idx_x] = pix_content/gaus_norm

    kernel_norm = np.sum(image_kernel)
    for idx_x1 in range(0,len(xaxis)-1):
        for idx_y1 in range(0,len(yaxis)-1):
            image_smooth[idx_y1,idx_x1] = 0.
            for idx_x2 in range(idx_x1-2*kernel_pix_size,idx_x1+2*kernel_pix_size):
                for idx_y2 in range(idx_y1-2*kernel_pix_size,idx_y1+2*kernel_pix_size):
                    if idx_x2<0: continue
                    if idx_y2<0: continue
                    if idx_x2>=len(xaxis)-1: continue
                    if idx_y2>=len(yaxis)-1: continue
                    old_content = image_data[idx_y2,idx_x2]
                    scale = image_kernel[central_bin_y+idx_y2-idx_y1,central_bin_x+idx_x2-idx_x1]
                    image_smooth[idx_y1,idx_x1] += old_content*scale
                    #image_smooth[idx_y1,idx_x1] += old_content*scale/kernel_norm

    for idx_x in range(0,len(xaxis)-1):
        for idx_y in range(0,len(yaxis)-1):
            image_data[idx_y,idx_x] = image_smooth[idx_y,idx_x]

class MyArray3D:

    def __init__(self,x_bins=10,start_x=0.,end_x=10.,y_bins=10,start_y=0.,end_y=10.,z_bins=10,start_z=0.,end_z=10.,overflow=True):
        array_shape = (x_bins,y_bins,z_bins)
        self.delta_x = (end_x-start_x)/float(x_bins)
        self.delta_y = (end_y-start_y)/float(y_bins)
        self.delta_z = (end_z-start_z)/float(z_bins)
        self.xaxis = np.zeros(array_shape[0]+1)
        self.yaxis = np.zeros(array_shape[1]+1)
        self.zaxis = np.zeros(array_shape[2]+1)
        self.waxis = np.zeros(array_shape)
        self.overflow = overflow
        for idx in range(0,len(self.xaxis)):
            self.xaxis[idx] = start_x + idx*self.delta_x
        for idx in range(0,len(self.yaxis)):
            self.yaxis[idx] = start_y + idx*self.delta_y
        for idx in range(0,len(self.zaxis)):
            self.zaxis[idx] = start_z + idx*self.delta_z

    def reset(self):
        for idx_x in range(0,len(self.xaxis)-1):
            for idx_y in range(0,len(self.yaxis)-1):
                for idx_z in range(0,len(self.zaxis)-1):
                    self.waxis[idx_x,idx_y,idx_z] = 0.

    def add(self, add_array, factor=1.):
        for idx_x in range(0,len(self.xaxis)-1):
            for idx_y in range(0,len(self.yaxis)-1):
                for idx_z in range(0,len(self.zaxis)-1):
                    self.waxis[idx_x,idx_y,idx_z] = self.waxis[idx_x,idx_y,idx_z]+add_array.waxis[idx_x,idx_y,idx_z]*factor

    def get_bin(self, value_x, value_y, value_z):
        key_idx_x = -1
        key_idx_y = -1
        key_idx_z = -1
        for idx_x in range(0,len(self.xaxis)-1):
            if self.xaxis[idx_x]<=value_x and self.xaxis[idx_x+1]>value_x:
                key_idx_x = idx_x
        for idx_y in range(0,len(self.yaxis)-1):
            if self.yaxis[idx_y]<=value_y and self.yaxis[idx_y+1]>value_y:
                key_idx_y = idx_y
        for idx_z in range(0,len(self.zaxis)-1):
            if self.zaxis[idx_z]<=value_z and self.zaxis[idx_z+1]>value_z:
                key_idx_z = idx_z
        if value_x>self.xaxis.max():
            key_idx_x = len(self.xaxis)-2
        if value_y>self.yaxis.max():
            key_idx_y = len(self.yaxis)-2
        if value_z>self.zaxis.max():
            key_idx_z = len(self.zaxis)-2
        return [key_idx_x,key_idx_y,key_idx_z]

    def fill(self, value_x, value_y, value_z, weight=1.):
        key_idx = self.get_bin(value_x, value_y, value_z)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        key_idx_z = key_idx[2]
        if key_idx_x==-1: 
            key_idx_x = 0
            if not self.overflow: weight = 0.
        if key_idx_y==-1: 
            key_idx_y = 0
            if not self.overflow: weight = 0.
        if key_idx_z==-1: 
            key_idx_z = 0
            if not self.overflow: weight = 0.
        if key_idx_x==len(self.xaxis): 
            key_idx_x = len(self.xaxis)-2
            if not self.overflow: weight = 0.
        if key_idx_y==len(self.yaxis): 
            key_idx_y = len(self.yaxis)-2
            if not self.overflow: weight = 0.
        if key_idx_z==len(self.zaxis): 
            key_idx_z = len(self.zaxis)-2
            if not self.overflow: weight = 0.
        self.waxis[key_idx_x,key_idx_y,key_idx_z] += 1.*weight
    
    def divide(self, add_array):
        for idx_x in range(0,len(self.xaxis)-1):
            for idx_y in range(0,len(self.yaxis)-1):
                for idx_z in range(0,len(self.zaxis)-1):
                    if add_array.waxis[idx_x,idx_y,idx_z]==0.:
                        self.waxis[idx_x,idx_y,idx_z] = 0.
                    else:
                        self.waxis[idx_x,idx_y,idx_z] = self.waxis[idx_x,idx_y,idx_z]/add_array.waxis[idx_x,idx_y,idx_z]

    def get_bin_center(self, idx_x, idx_y, idx_z):
        return [self.xaxis[idx_x]+0.5*self.delta_x,self.yaxis[idx_y]+0.5*self.delta_y,self.zaxis[idx_z]+0.5*self.delta_z]

    def get_bin_content(self, value_x, value_y, value_z):
        key_idx = self.get_bin(value_x, value_y, value_z)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        key_idx_z = key_idx[2]
        if key_idx_x==-1: 
            key_idx_x = 0
        if key_idx_y==-1: 
            key_idx_y = 0
        if key_idx_z==-1: 
            key_idx_z = 0
        if key_idx_x==len(self.xaxis): 
            key_idx_x = len(self.xaxis)-2
        if key_idx_y==len(self.yaxis): 
            key_idx_y = len(self.yaxis)-2
        if key_idx_z==len(self.zaxis): 
            key_idx_z = len(self.zaxis)-2
        return self.waxis[key_idx_x,key_idx_y,key_idx_z]

class MyArray1D:

    def __init__(self,x_bins=10,start_x=0.,end_x=10.,overflow=True):
        array_shape = (x_bins)
        self.delta_x = (end_x-start_x)/float(x_bins)
        self.xaxis = np.zeros(array_shape+1)
        self.waxis = np.zeros(array_shape)
        self.overflow = overflow
        for idx in range(0,len(self.xaxis)):
            self.xaxis[idx] = start_x + idx*self.delta_x

    def reset(self):
        for idx_x in range(0,len(self.xaxis)-1):
            self.waxis[idx_x] = 0.

    def add(self, add_array, factor=1.):
        for idx_x in range(0,len(self.xaxis)-1):
            self.waxis[idx_x] = self.waxis[idx_x]+add_array.waxis[idx_x]*factor

    def get_bin(self, value_x):
        key_idx_x = -1
        for idx_x in range(0,len(self.xaxis)-1):
            if self.xaxis[idx_x]<=value_x and self.xaxis[idx_x+1]>value_x:
                key_idx_x = idx_x
        if value_x>self.xaxis.max():
            key_idx_x = len(self.xaxis)-2
        return key_idx_x

    def fill(self, value_x, weight=1.):
        key_idx = self.get_bin(value_x)
        if key_idx==-1: 
            key_idx = 0
            if not self.overflow: weight = 0.
        if key_idx==len(self.xaxis): 
            key_idx = len(self.xaxis)-2
            if not self.overflow: weight = 0.
        self.waxis[key_idx] += 1.*weight
    
    def divide(self, add_array):
        for idx_x in range(0,len(self.xaxis)-1):
            if add_array.waxis[idx_x]==0.:
                self.waxis[idx_x] = 0.
            else:
                self.waxis[idx_x] = self.waxis[idx_x]/add_array.waxis[idx_x]

    def get_bin_center(self, idx_x):
        return self.xaxis[idx_x]+0.5*self.delta_x

    def get_bin_content(self, value_x):
        key_idx = self.get_bin(value_x)
        if key_idx==-1: 
            key_idx = 0
        if key_idx==len(self.xaxis): 
            key_idx = len(self.xaxis)-2
        return self.waxis[key_idx]

def GetGammaSources(tele_point_ra, tele_point_dec):

    bright_stars_coord = []
    inputFile = open(f'{smi_aux}/TeVCat_RaDec.txt')
    for line in inputFile:
        if line=='': continue
        line_split = line.split()
        if len(line_split)!=2: continue
        star_ra = float(line_split[0])
        star_dec = float(line_split[1])
        distance = pow(pow(star_ra-tele_point_ra,2)+pow(star_dec-tele_point_dec,2),0.5)
        if distance>2.: continue
        print (f'{line_split}')
        bright_stars_coord += [[star_ra,star_dec]]
    print (f'Found {len(bright_stars_coord)} Gamma-ray sources.')
    return bright_stars_coord

def GetBrightStars(tele_point_ra, tele_point_dec):

    brightness_cut = 6.0
    bright_stars_coord = []
    inputFile = open(f'{smi_aux}/Hipparcos_MAG8_1997.dat')
    for line in inputFile:
        if '#' in line: continue 
        if '*' in line: continue 
        if line=='': continue
        line_split = line.split()
        if len(line_split)!=5: continue
        star_ra = float(line_split[0])
        star_dec = float(line_split[1])
        star_brightness = float(line_split[3]) + float(line_split[4])
        distance = pow(pow(star_ra-tele_point_ra,2)+pow(star_dec-tele_point_dec,2),0.5)
        if distance>2.: continue
        print (f'{line_split}')
        if star_brightness<brightness_cut:
            bright_stars_coord += [[star_ra,star_dec]]

    print (f'Found {len(bright_stars_coord)} bright stars.')
    return bright_stars_coord


def CoincideWithBrightStars(ra, dec, bright_stars_coord):
    bright_star_radius_cut = 0.25
    isCoincident = False
    for star in range(0,len(bright_stars_coord)):
        star_ra = bright_stars_coord[star][0]
        star_dec = bright_stars_coord[star][1]
        distance = pow(pow(star_ra-ra,2)+pow(star_dec-dec,2),0.5)
        if distance>bright_star_radius_cut: continue
        isCoincident = True
    return isCoincident


def build_big_camera_matrix(smi_input,runlist,max_runs=1e10,is_on=True,specific_run=0):

    big_matrix = []
    for logE in range(0,logE_bins):
        big_matrix += [None]

    logE_axis = MyArray1D(x_bins=logE_bins,start_x=logE_start,end_x=logE_end)

    run_count = 0
    for run_number in runlist:
    
        rootfile_name = f'{smi_input}/{run_number}.anasum.root'
        print (rootfile_name)
        if not os.path.exists(rootfile_name):
            print (f'file does not exist.')
            continue
        if specific_run!=0:
            if specific_run!=run_number: 
                continue
        run_count += 1
    
        xyoff_map = []
        for logE in range(0,logE_bins):
            xyoff_map += [MyArray3D(x_bins=xoff_bins,start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins,start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    
        InputFile = ROOT.TFile(rootfile_name)

        TreeName = f'run_{run_number}/stereo/pointingDataReduced'
        TelTree = InputFile.Get(TreeName)
        TelTree.GetEntry(int(float(TelTree.GetEntries())/2.))
        TelRAJ2000 = TelTree.TelRAJ2000*180./np.pi
        TelDecJ2000 = TelTree.TelDecJ2000*180./np.pi
        bright_star_coord = GetBrightStars(TelRAJ2000,TelDecJ2000)
        gamma_source_coord = GetGammaSources(TelRAJ2000,TelDecJ2000)

        TreeName = f'run_{run_number}/stereo/DL3EventTree'
        EvtTree = InputFile.Get(TreeName)
        total_entries = EvtTree.GetEntries()
        print (f'total_entries = {total_entries}')
        for entry in range(0,total_entries):
            EvtTree.GetEntry(entry)
            Xoff = EvtTree.Xoff
            Yoff = EvtTree.Yoff
            Xderot = EvtTree.Xderot
            Yderot = EvtTree.Yderot
            MSCW = EvtTree.MSCW/MSCW_cut
            MSCL = EvtTree.MSCL/MSCL_cut
            GammaCut = pow(MSCW*MSCW+MSCL*MSCL,0.5)
            #MVA = EvtTree.MVA
            #GammaCut = (1.-MVA)/(1.-MVA_cut)
            Energy = EvtTree.Energy
            NImages = EvtTree.NImages
            EmissionHeight = EvtTree.EmissionHeight
            Xcore = EvtTree.XCore
            Ycore = EvtTree.YCore
            Roff = pow(Xoff*Xoff+Yoff*Yoff,0.5)
            Rcore = pow(Xcore*Xcore+Ycore*Ycore,0.5)
            logE = logE_axis.get_bin(np.log10(Energy))
            if logE<0: continue
            if logE>len(xyoff_map): continue
            if NImages<min_NImages: continue
            if EmissionHeight>max_EmissionHeight_cut: continue
            if EmissionHeight<min_EmissionHeight_cut: continue
            if Roff>max_Roff: continue
            if Rcore>max_Rcore: continue
            if Energy<min_Energy_cut: continue
            if Energy>max_Energy_cut: continue

            Xsky = TelRAJ2000 + Xderot
            Ysky = TelDecJ2000 + Yderot
            mirror_Xsky = TelRAJ2000 - Xderot
            mirror_Ysky = TelDecJ2000 - Yderot
            found_bright_star = CoincideWithBrightStars(Xsky, Ysky, bright_star_coord)
            found_gamma_source = CoincideWithBrightStars(Xsky, Ysky, gamma_source_coord)
            found_mirror_star = CoincideWithBrightStars(mirror_Xsky, mirror_Ysky, bright_star_coord)
            found_mirror_gamma_source = CoincideWithBrightStars(mirror_Xsky, mirror_Ysky, gamma_source_coord)

            if not is_on:
                if found_bright_star: continue
                if found_gamma_source: continue
                if found_mirror_star or found_mirror_gamma_source:
                    xyoff_map[logE].fill(-Xoff,-Yoff,GammaCut)

            xyoff_map[logE].fill(Xoff,Yoff,GammaCut)
    
        for logE in range(0,logE_bins):
            xyoff_map_1d = []
            for gcut in range(0,gcut_bins):
                for idx_x in range(0,xoff_bins):
                    for idx_y in range(0,yoff_bins):
                        xyoff_map_1d += [xyoff_map[logE].waxis[idx_x,idx_y,gcut]]
            if big_matrix[logE]==None:
                big_matrix[logE] = [xyoff_map_1d]
            else:
                big_matrix[logE] += [xyoff_map_1d]

        InputFile.Close()
        if run_count==max_runs: break

    return big_matrix

def build_skymap(smi_input,eigenvector_path,runlist,src_ra,src_dec,max_runs=1e10):

    skymap_size = 3.
    skymap_bins = 100
    xsky_start = src_ra-skymap_size
    xsky_end = src_ra+skymap_size
    ysky_start = src_dec-skymap_size
    ysky_end = src_dec+skymap_size

    print ('loading svd pickle data... ')
    input_filename = eigenvector_path
    big_eigenvectors = pickle.load(open(input_filename, "rb"))

    exposure_hours = 0.
    all_sky_map = []
    for logE in range(0,logE_bins):
        all_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

    data_xyoff_map = []
    fit_xyoff_map = []
    for logE in range(0,logE_bins):
        data_xyoff_map += [MyArray3D(x_bins=xoff_bins,start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins,start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
        fit_xyoff_map += [MyArray3D(x_bins=xoff_bins,start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins,start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

    for logE in range(0,logE_bins):
        print (f'big_eigenvectors[{logE}].shape = {big_eigenvectors[logE].shape}') 
        if matrix_rank>big_eigenvectors[logE].shape[0]:
            print (f'Not enough vectors. Break.')
            return exposure_hours, all_sky_map, data_xyoff_map, fit_xyoff_map

    logE_axis = MyArray1D(x_bins=logE_bins,start_x=logE_start,end_x=logE_end)

    run_count = 0
    for run_number in runlist:
    
        print (f'analyzing run {run_number}')
        rootfile_name = f'{smi_input}/{run_number}.anasum.root'
        print (rootfile_name)
        if not os.path.exists(rootfile_name):
            print (f'file does not exist.')
            continue
        run_count += 1
    
        print ('build big matrix...')
        big_on_matrix = build_big_camera_matrix(smi_input,runlist,max_runs=1e10,is_on=True,specific_run=run_number)
        
        ratio_xyoff_map = []
        for logE in range(0,logE_bins):
            ratio_xyoff_map += [MyArray3D(x_bins=xoff_bins,start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins,start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

        print ('fitting xyoff maps...')
        for logE in range(0,logE_bins):
            data_xyoff_map_1d = big_on_matrix[logE][0]
            init_params = [1e-3] * matrix_rank
            stepsize = [1e-3] * matrix_rank
            solution = minimize(
                cosmic_ray_like_chi2,
                x0=init_params,
                args=(big_eigenvectors[logE],data_xyoff_map_1d),
                method='L-BFGS-B',
                jac=None,
                options={'eps':stepsize,'ftol':0.001},
            )
            fit_params = solution['x']
            fit_xyoff_map_1d = big_eigenvectors[logE].T @ fit_params

            for gcut in range(0,gcut_bins):
                for idx_x in range(0,xoff_bins):
                    for idx_y in range(0,yoff_bins):
                        idx_1d = gcut*xoff_bins*yoff_bins + idx_x*yoff_bins + idx_y
                        data_xyoff_map[logE].waxis[idx_x,idx_y,gcut] += data_xyoff_map_1d[idx_1d]
                        fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] += fit_xyoff_map_1d[idx_1d]

            for gcut in range(0,gcut_bins):
                for idx_x in range(0,xoff_bins):
                    for idx_y in range(0,yoff_bins):
                        idx_1d = gcut*xoff_bins*yoff_bins + idx_x*yoff_bins + idx_y
                        glike_idx_1d = 0*xoff_bins*yoff_bins + idx_x*yoff_bins + idx_y
                        if fit_xyoff_map_1d[idx_1d]==0.: continue
                        ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = fit_xyoff_map_1d[glike_idx_1d]/fit_xyoff_map_1d[idx_1d]

            #for gcut in range(1,gcut_bins):
            #    avg_ratio = 0.
            #    for idx_x in range(0,xoff_bins):
            #        for idx_y in range(0,yoff_bins):
            #            avg_ratio += ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut]
            #    avg_ratio = avg_ratio/float(xoff_bins*yoff_bins)
            #    for idx_x in range(0,xoff_bins):
            #        for idx_y in range(0,yoff_bins):
            #            if ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut]<0.1*avg_ratio:
            #                ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = 0.1*avg_ratio
            #            if ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut]>10.*avg_ratio:
            #                ratio_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = 10.*avg_ratio


    
        InputFile = ROOT.TFile(rootfile_name)

        TreeName = f'run_{run_number}/stereo/pointingDataReduced'
        TelTree = InputFile.Get(TreeName)
        TelTree.GetEntry(int(float(TelTree.GetEntries())/2.))
        TelRAJ2000 = TelTree.TelRAJ2000*180./np.pi
        TelDecJ2000 = TelTree.TelDecJ2000*180./np.pi
        bright_star_coord = GetBrightStars(TelRAJ2000,TelDecJ2000)
        gamma_source_coord = GetGammaSources(TelRAJ2000,TelDecJ2000)

        TreeName = f'run_{run_number}/stereo/DL3EventTree'
        EvtTree = InputFile.Get(TreeName)
        total_entries = EvtTree.GetEntries()
        print (f'total_entries = {total_entries}')
        EvtTree.GetEntry(0)
        time_start = EvtTree.timeOfDay
        EvtTree.GetEntry(total_entries-1)
        time_end = EvtTree.timeOfDay
        exposure_hours += (time_end-time_start)/3600.
        for entry in range(0,total_entries):
            EvtTree.GetEntry(entry)
            Xoff = EvtTree.Xoff
            Yoff = EvtTree.Yoff
            Xderot = EvtTree.Xderot
            Yderot = EvtTree.Yderot
            MSCW = EvtTree.MSCW/MSCW_cut
            MSCL = EvtTree.MSCL/MSCL_cut
            GammaCut = pow(MSCW*MSCW+MSCL*MSCL,0.5)
            #MVA = EvtTree.MVA
            #GammaCut = (1.-MVA)/(1.-MVA_cut)
            Energy = EvtTree.Energy
            NImages = EvtTree.NImages
            EmissionHeight = EvtTree.EmissionHeight
            Xcore = EvtTree.XCore
            Ycore = EvtTree.YCore
            Roff = pow(Xoff*Xoff+Yoff*Yoff,0.5)
            Rcore = pow(Xcore*Xcore+Ycore*Ycore,0.5)
            logE = logE_axis.get_bin(np.log10(Energy))
            if logE<0: continue
            if logE>len(all_sky_map): continue
            if NImages<min_NImages: continue
            if EmissionHeight>max_EmissionHeight_cut: continue
            if EmissionHeight<min_EmissionHeight_cut: continue
            if Roff>max_Roff: continue
            if Rcore>max_Rcore: continue
            if Energy<min_Energy_cut: continue
            if Energy>max_Energy_cut: continue

            Xsky = TelRAJ2000 + Xderot
            Ysky = TelDecJ2000 + Yderot

            cr_correction = ratio_xyoff_map[logE].get_bin_content(Xoff,Yoff,GammaCut)

            all_sky_map[logE].fill(Xsky,Ysky,GammaCut,weight=cr_correction)


    return exposure_hours, all_sky_map, data_xyoff_map, fit_xyoff_map


def cosmic_ray_like_chi2(try_params,eigenvectors,xyoff_map):

    try_params = np.array(try_params)
    try_xyoff_map = eigenvectors.T @ try_params

    chi2 = 0.
    for gcut in range(1,gcut_bins):
        for idx_x in range(0,xoff_bins):
            for idx_y in range(0,yoff_bins):
                idx_1d = gcut*xoff_bins*yoff_bins + idx_x*yoff_bins + idx_y
                stat_err = 1.
                #stat_err = max(1.,pow(xyoff_map[idx_1d],0.5))
                chi2 += pow((try_xyoff_map[idx_1d]-xyoff_map[idx_1d])/stat_err,2)/float(gcut)

    return chi2

