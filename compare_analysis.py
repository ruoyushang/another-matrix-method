
import os, sys
import ROOT
import numpy as np
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
from common_functions import MyArray1D
from common_functions import MyArray3D

import common_functions

logE_threshold = common_functions.logE_threshold
logE_axis = common_functions.logE_axis
logE_nbins = common_functions.logE_nbins
logE_bins = common_functions.logE_bins
gcut_bins = common_functions.gcut_bins
xoff_bins = common_functions.xoff_bins
yoff_bins = common_functions.yoff_bins
xoff_start = common_functions.xoff_start
xoff_end = common_functions.xoff_end
yoff_start = common_functions.yoff_start
yoff_end = common_functions.yoff_end
gcut_start = common_functions.gcut_start
gcut_end = common_functions.gcut_end
ReadRunListFromFile = common_functions.ReadRunListFromFile
build_skymap = common_functions.build_skymap
smooth_image = common_functions.smooth_image
PlotSkyMap = common_functions.PlotSkyMap
make_flux_map = common_functions.make_flux_map
make_significance_map = common_functions.make_significance_map
DefineRegionOfInterest = common_functions.DefineRegionOfInterest
matrix_rank = common_functions.matrix_rank
skymap_size = common_functions.skymap_size
skymap_bins = common_functions.skymap_bins
significance_li_and_ma = common_functions.significance_li_and_ma
compute_camera_frame_power_spectrum = common_functions.compute_camera_frame_power_spectrum

skymap_bins = 80

smi_dir = os.environ.get("SMI_DIR")
smi_input = os.environ.get("SMI_INPUT")
#smi_output = os.environ.get("SMI_OUTPUT")
#smi_output = "/nevis/ged/data/rshang/smi_output/output_default"
smi_output = "/nevis/ged/data/rshang/smi_output/output_20250417"

ana_tag = []

#ana_tag += [['cr20_nbin9_fullspec16_fov10','b']]
#ana_tag += [['cr20_nbin9_fullspec16_fov15','b']]
#ana_tag += [['cr20_nbin9_fullspec16_free','b']]

#ana_tag += [['cr8_nbin7_init_free','initial']]
#ana_tag += [['cr8_nbin7_init_fov15','initial']]
#ana_tag += [['cr8_nbin7_init_fov10','initial']]
#ana_tag += [['cr8_nbin7_init_fov05','initial']]
#ana_tag += [['cr8_nbin7_init_fov03','initial']]
#ana_tag += [['cr8_nbin7_fullspec64_free','$k_{c}$=64']]

#ana_tag += [['cr8_nbin5_fullspec64_free','$k_{c}$=64']]

ana_tag += [['cr8_nbin7_fullspec1_free','$k_{c}$=1']]
ana_tag += [['cr8_nbin7_fullspec2_free','$k_{c}$=2']]
ana_tag += [['cr8_nbin7_fullspec4_free','$k_{c}$=4']]
ana_tag += [['cr8_nbin7_fullspec8_free','$k_{c}$=8']]
ana_tag += [['cr8_nbin7_fullspec16_free','$k_{c}$=16']]
ana_tag += [['cr8_nbin7_fullspec32_free','$k_{c}$=32']]
ana_tag += [['cr8_nbin7_fullspec64_free','$k_{c}$=64']]

#ana_tag += [['cr8_nbin0_fullspec64_free','energy-dep bins']]
#ana_tag += [['cr8_nbin1_fullspec64_free','$1\\times1$ bins']]
#ana_tag += [['cr8_nbin3_fullspec64_free','$3\\times3$ bins']]
#ana_tag += [['cr8_nbin5_fullspec64_free','$5\\times5$ bins']]
#ana_tag += [['cr8_nbin7_fullspec64_free','$7\\times7$ bins']]


onoff = 'OFF'

#exposure_per_group = 2.
#exposure_per_group = 5.
#exposure_per_group = 10.
#exposure_per_group = 20.
#exposure_per_group = 30.
exposure_per_group = 50.
#exposure_per_group = 100.
#exposure_per_group = 1000.
cr_qual_cut = 1e10
#cr_qual_cut = 230

min_elev = 40.
max_elev = 90.

#input_epoch = ['V4']
#input_epoch = ['V5']
#input_epoch = ['V6']
input_epoch = ['V4','V5','V6']
#input_epoch = ['V5','V6']

#demo_energy = logE_bins
#logE_low = 0
#logE_mid = logE_axis.get_bin(np.log10(0.3))+1
#logE_hig = logE_axis.get_bin(np.log10(1.7))+1
#demo_energy = [logE_bins[logE_low], logE_bins[logE_mid], logE_bins[logE_hig], logE_bins[len(logE_bins)-1]] # log10(E/TeV)
logE_low = logE_axis.get_bin(np.log10(0.2))+1
logE_hig = logE_axis.get_bin(np.log10(1.0))+1
demo_energy = [logE_bins[logE_low], logE_bins[logE_hig], logE_bins[len(logE_bins)-1]] # log10(E/TeV)
demoE_nbins = len(demo_energy) - 1 
demoE_axis = MyArray1D(x_bins=demo_energy)


input_sources = []
input_sources += [ ['1ES0647'               ,102.694 ,25.050 ] ]
input_sources += [ ['1ES1011'               ,153.767 ,49.434 ] ]
input_sources += [ ['1ES0414'               ,64.220  ,1.089  ] ]
input_sources += [ ['1ES0502'               ,76.983  ,67.623 ] ]
input_sources += [ ['1ES0229'               ,38.222  ,20.273 ] ]
input_sources += [ ['M82'                   ,148.970 ,69.679 ] ]
input_sources += [ ['3C264'                 ,176.271 ,19.606 ] ]
input_sources += [ ['BLLac'                 ,330.680 ,42.277 ] ]
input_sources += [ ['Draco'                 ,260.059 ,57.921 ] ]
input_sources += [ ['OJ287'                 ,133.705 ,20.100 ] ]
input_sources += [ ['H1426'                 ,217.136  ,42.673 ] ]
input_sources += [ ['NGC1275'               ,49.950  ,41.512 ] ]
input_sources += [ ['Segue1'                ,151.767 ,16.082 ] ]
input_sources += [ ['3C273'                 ,187.277 ,2.05   ] ]
input_sources += [ ['PG1553'                ,238.936 ,11.195 ] ]
input_sources += [ ['PKS1424'               ,216.750 ,23.783 ] ]
input_sources += [ ['RGB_J0710_p591'        ,107.61  ,59.15  ] ]
input_sources += [ ['UrsaMinor'             ,227.285 ,67.222 ] ]
input_sources += [ ['UrsaMajorII'           ,132.875 ,63.13  ] ]
input_sources += [ ['1ES1959_p650'          ,300.00 ,65.15 ] ]

src_keys = []
for src in input_sources:
    src_keys += [src[0]]

grp_data_map = []
grp_bkgd_map = []
grp_diff_map = []
for demoE in range(0,demoE_nbins):
    grp_data_map += [MyArray3D(x_bins=skymap_bins,start_x=xoff_start,end_x=xoff_end,y_bins=skymap_bins,start_y=yoff_start,end_y=yoff_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    grp_bkgd_map += [MyArray3D(x_bins=skymap_bins,start_x=xoff_start,end_x=xoff_end,y_bins=skymap_bins,start_y=yoff_start,end_y=yoff_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    grp_diff_map += [MyArray3D(x_bins=skymap_bins,start_x=xoff_start,end_x=xoff_end,y_bins=skymap_bins,start_y=yoff_start,end_y=yoff_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]


def get_analysis_data(ana):

    group_data = []
    current_exposure = 0.
    total_exposure = 0.
    expo_dict = dict.fromkeys(src_keys, 0.)  # Initializes all values to 0.
    nsb_dict = dict.fromkeys(src_keys, 0.)  # Initializes all values to 0.

    run_data = []
    for epoch in input_epoch:
        for src in input_sources:
    
            source_name = src[0]
    
            input_filename = f'{smi_output}/skymaps_{source_name}_{epoch}_{onoff}_{ana[0]}.pkl'
            if not os.path.exists(input_filename):
                continue
            analysis_result = pickle.load(open(input_filename, "rb"))
    
            for run in range(0,len(analysis_result)):
    
                run_info = analysis_result[run][0] 
                exposure = run_info[0]
                run_elev = run_info[1]
                run_azim = run_info[2]
                run_nsb = run_info[3]
                data_sky_map = analysis_result[run][2] 
                bkgd_sky_map = analysis_result[run][3] 
                syst_sky_map = analysis_result[run][4] 
                data_xyoff_map = analysis_result[run][5]
                bkgd_xyoff_map = analysis_result[run][6]

                for logE in range(0,logE_nbins):
                    syst_sky_map[logE].scale(1.0)

                logE_peak = 0
                bkgd_peak = 0.
                for logE in range(0,logE_nbins):
                    bkgd = np.sum(bkgd_xyoff_map[logE].waxis[:,:,:])
                    if bkgd>bkgd_peak:
                        bkgd_peak = bkgd
                        logE_peak = logE

                for logE in range(0,logE_nbins):
                    if logE<logE_peak+logE_threshold:
                        data_sky_map[logE].reset()
                        bkgd_sky_map[logE].reset()
                        syst_sky_map[logE].reset()
                        data_xyoff_map[logE].reset()
                        bkgd_xyoff_map[logE].reset()
    
                if run_azim>270.:
                    run_azim = run_azim-360.
    
                if exposure==0.: 
                    continue
                if run_elev<min_elev:
                    continue
                if run_elev>max_elev:
                    continue
                
                expo_dict[source_name] += exposure
                nsb_dict[source_name] += exposure * run_nsb

                total_exposure += exposure
                current_exposure += exposure
                run_data += [[source_name,exposure,run_elev,run_nsb,data_sky_map,bkgd_sky_map,syst_sky_map]]

                #if current_exposure>exposure_per_group or run==len(analysis_result)-1:
                if current_exposure>exposure_per_group:
                    current_exposure = 0.
                    run_data = []
                    group_data += [run_data]

    for src in range(0,len(src_keys)):
        src_name = src_keys[src]
        print (f"{src_name}, {expo_dict[src_name]:0.1f} hrs")
    print (f"Anaysis: {ana[0]}, total_exposure = {total_exposure:0.1f} hrs")

    return expo_dict, group_data

def GetRadialProfile(hist_skymap,roi_x,roi_y,roi_r,radial_bin_scale=0.1):

    deg2_to_sr =  3.046*1e-4
    pix_size = abs((hist_skymap.yaxis[1]-hist_skymap.yaxis[0])*(hist_skymap.xaxis[1]-hist_skymap.xaxis[0]))*deg2_to_sr
    bin_size = max(radial_bin_scale,1.*(hist_skymap.yaxis[1]-hist_skymap.yaxis[0]))
    radial_axis = MyArray1D(x_nbins=int(roi_r/bin_size),start_x=0.,end_x=roi_r)

    radius_array = []
    brightness_array = []
    brightness_err_array = []
    brightness_syst_array = []
    pixel_array = []
    for br in range(0,len(radial_axis.xaxis)-1):
        radius = 0.5*(radial_axis.xaxis[br]+radial_axis.xaxis[br+1])
        radius_array += [radius]
        brightness_array += [0.]
        brightness_err_array += [0.]
        brightness_syst_array += [0.]
        pixel_array += [0.]

    for br in range(0,len(radial_axis.xaxis)-1):
        radius = 0.5*(radial_axis.xaxis[br]+radial_axis.xaxis[br+1])
        for bx in range(0,len(hist_skymap.xaxis)-1):
            for by in range(0,len(hist_skymap.yaxis)-1):
                bin_ra = 0.5*(hist_skymap.xaxis[bx]+hist_skymap.xaxis[bx+1])
                bin_dec = 0.5*(hist_skymap.yaxis[by]+hist_skymap.yaxis[by+1])
                keep_event = False
                distance = pow(pow(bin_ra-roi_x,2) + pow(bin_dec-roi_y,2),0.5)
                if distance<radial_axis.xaxis[br+1] and distance>=radial_axis.xaxis[br]: 
                    keep_event = True
                if keep_event:
                    pixel_array[br] += 1.*pix_size
                    brightness_array[br] += hist_skymap.waxis[bx,by,0]
        #if pixel_array[br]==0.: continue
        #brightness_array[br] = brightness_array[br]/pixel_array[br]

    output_radius_array = []
    output_brightness_array = []
    for br in range(0,len(radial_axis.xaxis)-1):
        radius = radius_array[br]
        brightness = brightness_array[br]
        output_radius_array += [radius]
        output_brightness_array += [brightness]

    return np.array(output_radius_array), np.array(output_brightness_array)

def plot_normalization_error(ana_tag):
    

    radial_range = 1.8
    for demoE in range(0,demoE_nbins):
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 1.5 * 4.8))
        analysis_array = []
        error_array = []
        syst_array = []
        for ana in range(0,len(ana_tag)):
            expo_dict, ana_data = get_analysis_data(ana_tag[ana])

            analysis_error = []
            analysis_syst = []
            for grp  in range(0,len(ana_data)):
                grp_src_name = 0.
                grp_expo = 0.
                grp_elev = 0.
                grp_nsb = 0.
                grp_data_count = [0.] * demoE_nbins
                grp_bkgd_count = [0.] * demoE_nbins
                grp_syst_count = [0.] * demoE_nbins
                grp_data_map[demoE].reset()
                grp_bkgd_map[demoE].reset()

                for run in range(0,len(ana_data[grp])):
                    run_data = ana_data[grp][run]
                    src_name = run_data[0]
                    exposure = run_data[1]
                    run_elev = run_data[2]
                    run_nsb = run_data[3]
                    data_map = run_data[4]
                    bkgd_map = run_data[5]
                    syst_map = run_data[6]
                    grp_src_name = src_name
                    grp_expo += exposure
                    grp_elev += run_elev*exposure
                    grp_nsb += run_nsb*exposure

                    for logE in range(0,len(data_map)):
                        data_count = np.sum(data_map[logE].waxis[:,:,0])
                        bkgd_count = np.sum(bkgd_map[logE].waxis[:,:,0])
                        syst_count = np.sum(syst_map[logE].waxis[:,:,0])
                        if demoE != demoE_axis.get_bin(logE_bins[logE]): continue
                        grp_data_count[demoE] += data_count
                        grp_bkgd_count[demoE] += bkgd_count
                        grp_syst_count[demoE] += syst_count
                        grp_data_map[demoE].add(data_map[logE])
                        grp_bkgd_map[demoE].add(bkgd_map[logE])

                if grp_data_count[demoE]==0.: continue
                if grp_expo<exposure_per_group: continue

                analysis_error += [abs(grp_data_count[demoE]-grp_bkgd_count[demoE])/grp_data_count[demoE]]
                analysis_syst += [abs(grp_syst_count[demoE])/grp_data_count[demoE]]

            avg_analysis_error = np.mean(np.array(analysis_error))
            avg_analysis_syst = np.mean(np.array(analysis_syst))
            error_array += [avg_analysis_error]
            syst_array += [avg_analysis_syst]
            analysis_array += [ana_tag[ana][1]]

        ax.plot(error_array)
        ax.plot(syst_array)
        ax.set_title(f'E > {pow(10.,demo_energy[demoE]):0.2f} TeV')
        ax.set_xlabel('$k$ number of eigenvectors')
        ax.set_ylabel('$\\epsilon$ (%)')
        ax.set_xticks(np.arange(len(analysis_array)), labels=analysis_array)
        ax.set_yscale('log')
        fig.savefig(
            f"output_plots/normalization_error_demoE{demoE}.png", 
            dpi=300,
            bbox_inches="tight",
        )

        del fig
        del ax
        plt.close()


def plot_radial_profile(ana_tag):
    

    radial_range = 1.8
    for demoE in range(0,demoE_nbins):
        for ana in range(0,len(ana_tag)):
            expo_dict, ana_data = get_analysis_data(ana_tag[ana])

            avg_radius_array = []
            avg_significance_array = []
            fig, ax = plt.subplots(2, 1, figsize=(6.4, 1.5 * 4.8), gridspec_kw={'height_ratios': [2, 1]})
            for grp  in range(0,len(ana_data)):
                grp_src_name = 0.
                grp_expo = 0.
                grp_elev = 0.
                grp_nsb = 0.
                grp_data_count = [0.] * demoE_nbins
                grp_bkgd_count = [0.] * demoE_nbins
                grp_data_map[demoE].reset()
                grp_bkgd_map[demoE].reset()

                for run in range(0,len(ana_data[grp])):
                    run_data = ana_data[grp][run]
                    src_name = run_data[0]
                    exposure = run_data[1]
                    run_elev = run_data[2]
                    run_nsb = run_data[3]
                    data_map = run_data[4]
                    bkgd_map = run_data[5]
                    grp_src_name = src_name
                    grp_expo += exposure
                    grp_elev += run_elev*exposure
                    grp_nsb += run_nsb*exposure

                    for logE in range(0,len(data_map)):
                        data_count = np.sum(data_map[logE].waxis[:,:,0])
                        bkgd_count = np.sum(bkgd_map[logE].waxis[:,:,0])
                        if demoE != demoE_axis.get_bin(logE_bins[logE]): continue
                        grp_data_count[demoE] += data_count
                        grp_bkgd_count[demoE] += bkgd_count
                        grp_data_map[demoE].add(data_map[logE])
                        grp_bkgd_map[demoE].add(bkgd_map[logE])

                data_radius_array, data_profile_array = GetRadialProfile(grp_data_map[demoE],0.,0.,radial_range,radial_bin_scale=0.1)
                bkgd_radius_array, bkgd_profile_array = GetRadialProfile(grp_bkgd_map[demoE],0.,0.,radial_range,radial_bin_scale=0.1)

                if np.sum(data_profile_array)==0.: continue
                if grp_expo<exposure_per_group: continue

                significance_array = np.zeros_like(data_profile_array)
                for b in range(0,len(data_profile_array)):
                    data = data_profile_array[b]
                    bkgd = bkgd_profile_array[b]
                    significance = significance_li_and_ma(data, bkgd, 0.)
                    significance_array[b] = significance

                ax[0].plot(data_radius_array, significance_array, alpha=0.3)

                avg_radius_array += [data_radius_array]
                avg_significance_array += [np.abs(significance_array)]

            avg_radius_array = np.array(avg_radius_array[0])
            avg_significance_array = np.array(avg_significance_array)
            avg_significance_array = np.mean(avg_significance_array, axis=0)
            ax[1].plot(avg_radius_array, avg_significance_array)

            ax[0].set_title(f'E > {pow(10.,demo_energy[demoE]):0.2f} TeV')
            ax[0].set_ylabel('error significance [$\\sigma$]')
            ax[0].set_ylim(-10., 10.)
            ax[1].set_ylabel('avg. significance (absolute)')
            ax[1].set_xlabel('angular distance to camera center [deg]')
            ax[1].set_ylim(0., 4.)
            if 'fov' in ana_tag[ana][0]:
                ring_radius = float(ana_tag[ana][0].split('fov')[1])/10.
                ax[0].axvspan(ring_radius, min(radial_range,2.*ring_radius), color='gray', alpha=0.5)
                ax[1].axvspan(ring_radius, min(radial_range,2.*ring_radius), color='gray', alpha=0.5)
            fig.savefig(
                f"output_plots/radial_profile_{ana_tag[ana][0]}_demoE{demoE}.png", 
                dpi=300,
                bbox_inches="tight",
            )

            del fig
            del ax
            plt.close()


plot_normalization_error(ana_tag)
#plot_radial_profile(ana_tag)

