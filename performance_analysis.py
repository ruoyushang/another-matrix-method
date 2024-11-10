
import os, sys
import ROOT
import numpy as np
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
from common_functions import MyArray1D
from common_functions import MyArray3D

import common_functions

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
GetRadialProfile = common_functions.GetRadialProfile
matrix_rank = common_functions.matrix_rank
skymap_size = common_functions.skymap_size
skymap_bins = common_functions.skymap_bins
significance_li_and_ma = common_functions.significance_li_and_ma


smi_dir = os.environ.get("SMI_DIR")
smi_input = os.environ.get("SMI_INPUT")
#smi_output = os.environ.get("SMI_OUTPUT")
#smi_output = "/nevis/ged/data/rshang/smi_output/output_test_4"
smi_output = "/nevis/ged/data/rshang/smi_output/output_7x7"
#smi_output = "/nevis/ged/data/rshang/smi_output/output_default"

ana_tag = []
#ana_tag += [['binspec','r']]
#ana_tag += [['fullspec','b']]
#ana_tag += [['init','r']]

#ana_tag += [['rank1','r']]
#ana_tag += [['rank2','w']]
ana_tag += [['rank4','b']]
ana_tag += [['rank8','b']]
#ana_tag += [['rank16','b']]
#ana_tag += [['rank32','b']]
#ana_tag += [['rank64','b']]

onoff = 'OFF'

exposure_per_group = 2.
#exposure_per_group = 4.
#exposure_per_group = 10.
#exposure_per_group = 20.
#exposure_per_group = 50.
#exposure_per_group = 100.
cr_qual_cut = 1e10
#cr_qual_cut = 230

min_elev = 20.
#min_elev = 55.

#input_epoch = ['V4']
#input_epoch = ['V5']
#input_epoch = ['V6']
input_epoch = ['V4','V5','V6']

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
#input_sources += [ ['CrabNebula_elev_80_90' ,83.633  ,22.014 ] ]
#input_sources += [ ['CrabNebula_elev_70_80' ,83.633  ,22.014 ] ]
#input_sources += [ ['CrabNebula_elev_60_70' ,83.633  ,22.014 ] ]
#input_sources += [ ['CrabNebula_elev_50_60' ,83.633  ,22.014 ] ]
#input_sources += [ ['CrabNebula_elev_40_50' ,83.633  ,22.014 ] ]
#input_sources += [ ['CrabNebula_elev_30_40' ,83.633  ,22.014 ] ]
#input_sources += [ ['CrabNebula_1p0wobble' ,83.633  ,22.014 ] ]
#input_sources += [ ['CrabNebula_1p5wobble' ,83.633  ,22.014 ] ]


print (f"smi_output = {smi_output}")
print (f"exposure_per_group = {exposure_per_group} hrs")
analysis_data = []
for ana in range(0,len(ana_tag)):

    group_data = []
    current_exposure = 0.
    total_exposure = 0.
    
    run_data = []
    for epoch in input_epoch:
        for src in input_sources:
    
            source_name = src[0]
    
            input_filename = f'{smi_output}/skymaps_{source_name}_{epoch}_{onoff}_{ana_tag[ana][0]}.pkl'
            if not os.path.exists(input_filename):
                continue
            analysis_result = pickle.load(open(input_filename, "rb"))
    
            for run in range(0,len(analysis_result)):
    
                run_info = analysis_result[run][0] 
                exposure = run_info[0]
                run_elev = run_info[1]
                run_azim = run_info[2]
                truth_params = run_info[3]
                fit_params = run_info[4]
                run_nsb = run_info[5]
                data_sky_map = analysis_result[run][2] 
                bkgd_sky_map = analysis_result[run][3] 
                syst_sky_map = analysis_result[run][4] 
                data_xyoff_map = analysis_result[run][5]
                bkgd_xyoff_map = analysis_result[run][6]
    
                if run_azim>270.:
                    run_azim = run_azim-360.
    
                if exposure==0.: 
                    continue
                if run_elev<min_elev:
                    continue
    
                total_exposure += exposure
                current_exposure += exposure
                run_data += [[exposure,run_elev,run_nsb,data_xyoff_map,bkgd_xyoff_map]] # gamma-ray sources are not filtered in xyoff maps
                #run_data += [[exposure,run_elev,run_nsb,data_sky_map,bkgd_sky_map]]

                #if current_exposure>exposure_per_group or run==len(analysis_result)-1:
                if current_exposure>exposure_per_group:
                    current_exposure = 0.
                    run_data = []
                    group_data += [run_data]

    analysis_data += [group_data]
    print (f"Anaysis: {ana_tag[ana][0]}, total_exposure = {total_exposure:0.1f} hrs")
    
grp_data_map = []
grp_bkgd_map = []
for logE in range(0,logE_nbins):
    grp_data_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    grp_bkgd_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]

list_elev = []
list_nsb = []
list_data_count = []
list_bkgd_count = []
list_significance = []
for ana in range(0,len(analysis_data)):
    ana_elev = []
    ana_nsb = []
    ana_data_count = []
    ana_bkgd_count = []
    ana_significance = []
    for grp  in range(0,len(analysis_data[ana])):
        grp_expo = 0.
        grp_elev = 0.
        grp_nsb = 0.
        grp_data_count = np.zeros(len(logE_bins)-1)
        grp_bkgd_count = np.zeros(len(logE_bins)-1)
        grp_significance = [[] for i in range(0,len(logE_bins)-1)]
        for logE in range(0,len(logE_bins)-1):
            grp_data_map[logE].reset()
            grp_bkgd_map[logE].reset()
        for run in range(0,len(analysis_data[ana][grp])):
            run_data = analysis_data[ana][grp][run]
            exposure = run_data[0]
            run_elev = run_data[1]
            run_nsb = run_data[2]
            data_map = run_data[3]
            bkgd_map = run_data[4]
            grp_expo += exposure
            grp_elev += run_elev*exposure
            grp_nsb += run_nsb*exposure
            for logE in range(0,len(data_map)):
                data_count = np.sum(data_map[logE].waxis[:,:,0])
                bkgd_count = np.sum(bkgd_map[logE].waxis[:,:,0])
                grp_data_count[logE] += data_count
                grp_bkgd_count[logE] += bkgd_count
                grp_data_map[logE].add(data_map[logE])
                grp_bkgd_map[logE].add(bkgd_map[logE])
        if grp_expo==0.: continue
        grp_elev = grp_elev/grp_expo
        grp_nsb = grp_nsb/grp_expo
        ana_elev += [grp_elev]
        ana_nsb += [grp_nsb]
        ana_data_count += [grp_data_count]
        ana_bkgd_count += [grp_bkgd_count]
        for logE in range(0,len(data_map)):
            nbins_x = len(grp_data_map[logE].xaxis)-1
            nbins_y = len(grp_data_map[logE].yaxis)-1
            for binx in range (0,nbins_x):
                for biny in range (0,nbins_y):
                    data = grp_data_map[logE].waxis[binx,biny,0]
                    bkgd = grp_bkgd_map[logE].waxis[binx,biny,0]
                    significance = significance_li_and_ma(data, bkgd, 0.)
                    if data>0.:
                        grp_significance[logE] += [significance]
        ana_significance += [grp_significance]
    list_elev += [ana_elev]
    list_nsb += [ana_nsb]
    list_data_count += [ana_data_count]
    list_bkgd_count += [ana_bkgd_count]
    list_significance += [ana_significance]

for logE in range(0,logE_nbins):
    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = 'Elevation [deg]'
    label_y = 'Error significance'
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    for ana in range(0,len(ana_tag)):
        if ana_tag[ana][1]=='w': continue
        plot_elev = []
        plot_error_significance = []
        for grp in range(0,len(list_elev[ana])):
            elev = list_elev[ana][grp]
            data = list_data_count[ana][grp][logE]
            bkgd = list_bkgd_count[ana][grp][logE]
            plot_elev += [elev]
            plot_error_significance += [abs(significance_li_and_ma(data,bkgd,0.))]
        ax.scatter(plot_elev,plot_error_significance,color=ana_tag[ana][1],alpha=0.3, label=f"{ana_tag[ana][0].strip('rank')} eigenvectors")
        #ax.scatter(plot_elev,plot_error_significance,alpha=0.3, label=f"{ana_tag[ana][0].strip('rank')} eigenvectors")
    ax.legend(loc='best')
    fig.savefig(f'output_plots/fov_error_significance_vs_elev_logE{logE}.png',bbox_inches='tight')
    del fig
    del ax
    plt.close()

for logE in range(0,logE_nbins):
    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = 'Pedestal variance [p.e.]'
    label_y = 'Camera-wise error significance'
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    for ana in range(0,len(ana_tag)):
        if ana_tag[ana][1]=='w': continue
        plot_nsb = []
        plot_error_significance = []
        for grp in range(0,len(list_nsb[ana])):
            nsb = list_nsb[ana][grp]
            data = list_data_count[ana][grp][logE]
            bkgd = list_bkgd_count[ana][grp][logE]
            plot_nsb += [nsb]
            plot_error_significance += [abs(significance_li_and_ma(data,bkgd,0.))]
        ax.scatter(plot_nsb,plot_error_significance,color=ana_tag[ana][1],alpha=0.3, label=f"{ana_tag[ana][0].strip('rank')} eigenvectors")
        #ax.scatter(plot_nsb,plot_error_significance,alpha=0.3, label=f"{ana_tag[ana][0].strip('rank')} eigenvectors")
    ax.legend(loc='best')
    fig.savefig(f'output_plots/fov_error_significance_vs_nsb_logE{logE}.png',bbox_inches='tight')
    del fig
    del ax
    plt.close()

for logE in range(0,logE_nbins):
    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 3.8
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = 'Elevation [deg]'
    label_y = 'Pixel-wise error significance'
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    for ana in range(0,len(ana_tag)):
        if ana_tag[ana][1]=='w': continue
        plot_elev = []
        plot_error_significance = []
        for grp in range(0,len(list_elev[ana])):
            elev = list_elev[ana][grp]
            for pix in range(0,len(list_significance[ana][grp][logE])):
                significance = list_significance[ana][grp][logE][pix]
                plot_elev += [elev]
                plot_error_significance += [abs(significance)]
        kc = ana_tag[ana][0].strip('rank')
        scatter_color = 'k'
        if ana_tag[ana][1]=='b':
            scatter_color = 'skyblue'
        if ana_tag[ana][1]=='r':
            scatter_color = 'salmon'
        ax.scatter(plot_elev,plot_error_significance,color=scatter_color,alpha=0.05)

        n_bins = 10
        elev_min = np.min(plot_elev)
        elev_max = np.max(plot_elev)
        elev_delta = (elev_max-elev_min)/float(n_bins)
        elev_axis = []
        significance_axis = []
        for b in range(0,n_bins):
            elev_low = elev_min + b*elev_delta
            elev_up = elev_min + (b+1)*elev_delta
            elev_axis += [0.5*(elev_low+elev_up)]
            avg_significance = 0.
            n_entries = 0.
            for entry in range(0,len(plot_elev)):
                if plot_elev[entry]<elev_low: continue
                if plot_elev[entry]>elev_up: continue
                avg_significance += plot_error_significance[entry]
                n_entries += 1.
            if n_entries>0.:
                avg_significance = avg_significance/n_entries
            significance_axis += [avg_significance]
        ax.plot(elev_axis,significance_axis,color=ana_tag[ana][1],label='$k_{c}$='+f'{kc}')

    ax.legend(loc='best')
    E_min = pow(10.,logE_bins[logE])
    E_max = pow(10.,logE_bins[logE+1])
    ax.set_title(f'E = {E_min:0.2f} - {E_max:0.2f} TeV')
    fig.savefig(f'output_plots/pix_error_significance_vs_elev_logE{logE}.png',bbox_inches='tight')
    del fig
    del ax
    plt.close()

for logE in range(0,logE_nbins):
    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 3.8
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = 'Pedestal variance [p.e.]'
    label_y = 'Pixel-wise error significance'
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    for ana in range(0,len(ana_tag)):
        if ana_tag[ana][1]=='w': continue
        plot_nsb = []
        plot_error_significance = []
        for grp in range(0,len(list_nsb[ana])):
            nsb = list_nsb[ana][grp]
            for pix in range(0,len(list_significance[ana][grp][logE])):
                significance = list_significance[ana][grp][logE][pix]
                plot_nsb += [nsb]
                plot_error_significance += [abs(significance)]
        kc = ana_tag[ana][0].strip('rank')
        n_entries = len(plot_nsb)
        scatter_color = 'k'
        if ana_tag[ana][1]=='b':
            scatter_color = 'skyblue'
        if ana_tag[ana][1]=='r':
            scatter_color = 'salmon'
        ax.scatter(plot_nsb,plot_error_significance,color=scatter_color,alpha=0.05)

        n_bins = 10
        nsb_min = np.min(plot_nsb)
        nsb_max = np.max(plot_nsb)
        nsb_delta = (nsb_max-nsb_min)/float(n_bins)
        nsb_axis = []
        significance_axis = []
        for b in range(0,n_bins):
            nsb_low = nsb_min + b*nsb_delta
            nsb_up = nsb_min + (b+1)*nsb_delta
            nsb_axis += [0.5*(nsb_low+nsb_up)]
            avg_significance = 0.
            n_entries = 0.
            for entry in range(0,len(plot_nsb)):
                if plot_nsb[entry]<nsb_low: continue
                if plot_nsb[entry]>nsb_up: continue
                avg_significance += plot_error_significance[entry]
                n_entries += 1.
            if n_entries>0.:
                avg_significance = avg_significance/n_entries
            significance_axis += [avg_significance]
        ax.plot(nsb_axis,significance_axis,color=ana_tag[ana][1],label='$k_{c}$='+f'{kc}')

    ax.legend(loc='best')
    E_min = pow(10.,logE_bins[logE])
    E_max = pow(10.,logE_bins[logE+1])
    ax.set_title(f'E = {E_min:0.2f} - {E_max:0.2f} TeV')
    fig.savefig(f'output_plots/pix_error_significance_vs_nsb_logE{logE}.png',bbox_inches='tight')
    del fig
    del ax
    plt.close()

for logE in range(0,logE_nbins):

    hist_range = 10.
    hist_bins = 40

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 3.8
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = 'Significance'
    label_y = 'Entries'
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    for ana in range(0,len(ana_tag)):
        if ana_tag[ana][1]=='w': continue
        plot_error_significance = []
        for grp in range(0,len(list_elev[ana])):
            for pix in range(0,len(list_significance[ana][grp][logE])):
                significance = list_significance[ana][grp][logE][pix]
                if np.isnan(significance): continue
                plot_error_significance += [(significance)]
        hist_significance, bin_edges = np.histogram(plot_error_significance,bins=hist_bins,range=(-hist_range,hist_range))
        mean = np.mean(np.array(plot_error_significance))
        rms = np.sqrt(np.mean(np.square(np.array(plot_error_significance))))
        kc = ana_tag[ana][0].strip('rank')
        ax.hist(plot_error_significance,bin_edges,histtype='step',density=True,facecolor=ana_tag[ana][1],label='$k_{c}$='+f'{kc}, $\mu$={mean:0.2f}, $\sigma$={rms:0.2f}')
        #ax.hist(plot_error_significance,bin_edges,histtype='step',density=True,label=f'{ana_tag[ana][0]}')

    hist_binsize = 2.*hist_range/float(hist_bins)
    significance_axis = np.arange(-5., 5., 0.01)
    #total_entries = len(list_significance[0][logE])
    total_entries = 1.
    #normal_dist = hist_binsize*total_entries/pow(2.*np.pi,0.5)*np.exp(-significance_axis*significance_axis/2.)
    normal_dist = hist_binsize*total_entries*np.exp(-significance_axis*significance_axis/2.)
    ax.plot(significance_axis,normal_dist,color='k',linestyle='dashed')

    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.set_ylim(bottom=1e-3)
    E_min = pow(10.,logE_bins[logE])
    E_max = pow(10.,logE_bins[logE+1])
    ax.set_title(f'E = {E_min:0.2f} - {E_max:0.2f} TeV')
    fig.savefig(f'output_plots/error_significance_distribution_logE{logE}.png',bbox_inches='tight')
    del fig
    del ax
    plt.close()


for logE in range(0,logE_nbins):

    ana_axis = []
    ana_axis_label = []
    for ana in range(0,len(ana_tag)):
        ana_axis += [ana]
        ana_axis_label += [ana_tag[ana][0].strip("rank")]

    syst_err_axis = []
    stat_err_axis = []
    print ("====================================================================================================")
    for ana in range(0,len(ana_tag)):
        ana_data = 0.
        ana_bkgd = 0.
        ana_diff= 0.
        n_groups = float(len(list_elev[ana]))
        for grp in range(0,len(list_elev[ana])):
            data = list_data_count[ana][grp][logE]
            bkgd = list_bkgd_count[ana][grp][logE]
            ana_data += data/n_groups
            ana_bkgd += bkgd/n_groups
            ana_diff += pow(data-bkgd,2)/n_groups
        ana_diff = pow(ana_diff,0.5)
        syst_err = 0.
        stat_err = 0.
        if ana_data>0.:
            syst_err = ana_diff/ana_data
            stat_err = 1./pow(ana_data,0.5)
        syst_err_axis += [syst_err*100.]
        stat_err_axis += [pow(stat_err*stat_err+syst_err*syst_err/n_groups,0.5)*100.]
        print (f"E = {pow(10.,logE_bins[logE]):0.3f} TeV, {ana_tag[ana][0]}, error = {syst_err*100.:0.2f} +/- {stat_err*100.:0.2f} +/- {syst_err/pow(n_groups,0.5)*100.:0.2f} % ({syst_err/stat_err:0.2f} sigma)")

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 3.8
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = '$k$ number of eigenvectors'
    label_y = '$\epsilon$ (%)'
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    E_min = pow(10.,logE_bins[logE])
    E_max = pow(10.,logE_bins[logE+1])
    ax.errorbar(ana_axis,syst_err_axis,yerr=stat_err_axis,color='k',marker='.',ls='solid')
    ax.set_xticks(np.arange(len(ana_axis_label)), labels=ana_axis_label)
    ax.set_title(f'E = {E_min:0.2f} - {E_max:0.2f} TeV')
    fig.savefig(f'output_plots/ana_syst_err_logE{logE}.png',bbox_inches='tight')
    del fig
    del ax
    plt.close()

for ana in range(0,len(ana_tag)):
    n_groups = float(len(list_elev[ana]))
    print (f"Anaysis: {ana_tag[ana][0]}, n_groups = {n_groups}")

