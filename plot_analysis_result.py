
import os, sys
import ROOT
import numpy as np
import pickle
from matplotlib import pyplot as plt
from common_functions import MyArray1D
from common_functions import MyArray3D

import common_functions

logE_min = common_functions.logE_min
logE_max = common_functions.logE_max
logE_axis = common_functions.logE_axis
logE_nbins = common_functions.logE_nbins
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
PrintFluxCalibration = common_functions.PrintFluxCalibration
GetRadialProfile = common_functions.GetRadialProfile
matrix_rank = common_functions.matrix_rank
skymap_size = common_functions.skymap_size
skymap_bins = common_functions.skymap_bins

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

smi_input = os.environ.get("SMI_INPUT")
smi_output = os.environ.get("SMI_OUTPUT")
smi_dir = os.environ.get("SMI_DIR")

ana_tag = 'scale1'

source_name = sys.argv[1]
src_ra = float(sys.argv[2])
src_dec = float(sys.argv[3])
#input_epoch = ['V4']
#input_epoch = ['V5']
#input_epoch = ['V6']
input_epoch = ['V4','V5','V6']

xsky_start = src_ra+skymap_size
xsky_end = src_ra-skymap_size
ysky_start = src_dec-skymap_size
ysky_end = src_dec+skymap_size

roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r = DefineRegionOfInterest(source_name,src_ra,src_dec)

total_exposure = 0.
list_run_elev = []
list_run_azim = []
list_truth_params = []
list_fit_params = []
list_sr_qual = []
list_cr_qual = []
sum_data_sky_map = []
sum_bkgd_sky_map = []
sum_data_sky_map_smooth = []
sum_bkgd_sky_map_smooth = []
sum_excess_sky_map_smooth = []
sum_significance_sky_map = []
sum_flux_sky_map = []
sum_flux_err_sky_map = []
for logE in range(0,logE_nbins):
    sum_data_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_bkgd_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    sum_data_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_bkgd_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    sum_excess_sky_map_smooth += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_significance_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_flux_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
    sum_flux_err_sky_map += [MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)]
sum_data_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_bkgd_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)
sum_significance_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_excess_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)
sum_flux_err_sky_map_allE = MyArray3D(x_bins=skymap_bins,start_x=xsky_start,end_x=xsky_end,y_bins=skymap_bins,start_y=ysky_start,end_y=ysky_end,z_bins=1,start_z=gcut_start,end_z=gcut_end)

sum_data_xyoff_map = []
sum_fit_xyoff_map = []
sum_err_xyoff_map = []
sum_init_err_xyoff_map = []
for logE in range(0,logE_nbins):
    sum_data_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    sum_fit_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    sum_err_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]
    sum_init_err_xyoff_map += [MyArray3D(x_bins=xoff_bins[logE],start_x=xoff_start,end_x=xoff_end,y_bins=yoff_bins[logE],start_y=yoff_start,end_y=yoff_end,z_bins=gcut_bins,start_z=gcut_start,end_z=gcut_end)]

for epoch in input_epoch:

    input_filename = f'{smi_output}/skymaps_{source_name}_{epoch}_{ana_tag}.pkl'
    print (f'reading {input_filename}...')
    if not os.path.exists(input_filename):
        print (f'{input_filename} does not exist.')
        continue
    analysis_result = pickle.load(open(input_filename, "rb"))
    
    for run in range(0,len(analysis_result)):

        run_info = analysis_result[run][0] 
        exposure = run_info[0]
        run_elev = run_info[1]
        run_azim = run_info[2]
        truth_params = run_info[3]
        fit_params = run_info[4]
        sr_qual = run_info[5]
        cr_qual = run_info[6]

        if run_azim>270.:
            run_azim = run_azim-360.

        #is_good_run = True
        #for logE in range(0,3):
        #    if cr_qual[logE]>0.6:
        #        is_good_run = False
        #if not is_good_run: 
        #    print (f'bad fitting. reject the run.')
        #    continue

        data_sky_map = analysis_result[run][1] 
        bkgd_sky_map = analysis_result[run][2] 
        data_xyoff_map = analysis_result[run][3]
        fit_xyoff_map = analysis_result[run][4]

        total_exposure += exposure
        list_run_elev += [run_elev]
        list_run_azim += [run_azim]
        list_truth_params += [truth_params]
        list_fit_params += [fit_params]
        list_sr_qual += [sr_qual]
        list_cr_qual += [cr_qual]

        for logE in range(0,logE_nbins):
            if logE<logE_min: continue
            if logE>logE_max: continue
            sum_data_sky_map[logE].add(data_sky_map[logE])
            sum_bkgd_sky_map[logE].add(bkgd_sky_map[logE])
            sum_data_xyoff_map[logE].add(data_xyoff_map[logE])
            sum_fit_xyoff_map[logE].add(fit_xyoff_map[logE])
    
for logE in range(0,logE_nbins):
    data_integral = 0.
    model_integral = 0.
    for idx_x in range(0,xoff_bins[logE]):
        for idx_y in range(0,yoff_bins[logE]):
            data_integral += sum_data_xyoff_map[logE].waxis[idx_x,idx_y,0] 
            model_integral += sum_data_xyoff_map[logE].waxis[idx_x,idx_y,1] 
    if model_integral==0.: continue
    for idx_x in range(0,xoff_bins[logE]):
        for idx_y in range(0,yoff_bins[logE]):
            data = sum_data_xyoff_map[logE].waxis[idx_x,idx_y,0] 
            model = sum_data_xyoff_map[logE].waxis[idx_x,idx_y,1]*data_integral/model_integral 
            data_err = max(1.,pow(data,0.5))
            sum_init_err_xyoff_map[logE].waxis[idx_x,idx_y,0] = (data-model)/data_err

for logE in range(0,logE_nbins):
    for gcut in range(0,gcut_bins):
        for idx_x in range(0,xoff_bins[logE]):
            for idx_y in range(0,yoff_bins[logE]):
                data = sum_data_xyoff_map[logE].waxis[idx_x,idx_y,gcut] 
                model = sum_fit_xyoff_map[logE].waxis[idx_x,idx_y,gcut] 
                data_err = max(1.,pow(data,0.5))
                sum_err_xyoff_map[logE].waxis[idx_x,idx_y,gcut] = (data-model)/data_err

for logE in range(0,logE_nbins):
    sum_data_xyoff_map[logE].scale(1./total_exposure)
    sum_fit_xyoff_map[logE].scale(1./total_exposure)

for logE in range(0,logE_nbins):
    sum_data_sky_map_smooth[logE].reset()
    sum_bkgd_sky_map_smooth[logE].reset()
    sum_excess_sky_map_smooth[logE].reset()
    sum_data_sky_map_smooth[logE].add(sum_data_sky_map[logE])
    sum_bkgd_sky_map_smooth[logE].add(sum_bkgd_sky_map[logE])
    smooth_image(sum_bkgd_sky_map_smooth[logE].waxis[:,:,0],sum_bkgd_sky_map_smooth[logE].xaxis,sum_bkgd_sky_map_smooth[logE].yaxis,kernel_radius=0.14)
    for gcut in range(1,gcut_bins):
        smooth_image(sum_bkgd_sky_map_smooth[logE].waxis[:,:,gcut],sum_bkgd_sky_map_smooth[logE].xaxis,sum_bkgd_sky_map_smooth[logE].yaxis,kernel_radius=0.07)
    smooth_image(sum_data_sky_map_smooth[logE].waxis[:,:,0],sum_data_sky_map_smooth[logE].xaxis,sum_data_sky_map_smooth[logE].yaxis,kernel_radius=0.07)
    sum_data_sky_map_allE.add(sum_data_sky_map_smooth[logE])
    sum_bkgd_sky_map_allE.add(sum_bkgd_sky_map_smooth[logE])

print ('=================================================================================')
for logE in range(0,logE_nbins):

    gcut_weight = []
    for gcut in range(0,gcut_bins):
        gcut_weight += [pow(np.sum(sum_bkgd_sky_map[logE].waxis[:,:,gcut]),1.0)]

    data_sum = np.sum(sum_data_sky_map[logE].waxis[:,:,0])
    bkgd_sum = 0.
    gcut_norm = 0.
    for gcut in range(1,gcut_bins):
        gcut_norm += gcut_weight[gcut]
    for gcut in range(1,gcut_bins):
        bkgd_sum += gcut_weight[gcut]/gcut_norm*np.sum(sum_bkgd_sky_map[logE].waxis[:,:,gcut])

    error = 0.
    stat_error = 0.
    if data_sum>0.:
        error = 100.*(data_sum-bkgd_sum)/data_sum
        stat_error = 100.*pow(data_sum,0.5)/data_sum
    print (f'logE = {logE}, data_sum = {data_sum}, bkgd_sum = {bkgd_sum:0.1f}, error = {error:0.1f} +/- {stat_error:0.1f} %')

for logE in range(0,logE_nbins):
    make_significance_map(sum_data_sky_map_smooth[logE],sum_bkgd_sky_map_smooth[logE],sum_significance_sky_map[logE],sum_excess_sky_map_smooth[logE])
make_significance_map(sum_data_sky_map_allE,sum_bkgd_sky_map_allE,sum_significance_sky_map_allE,sum_excess_sky_map_allE)

for logE in range(0,logE_nbins):

    avg_energy = 0.5*(pow(10.,logE_axis.xaxis[logE])+pow(10.,logE_axis.xaxis[logE+1]))
    delta_energy = 0.5*(pow(10.,logE_axis.xaxis[logE+1])-pow(10.,logE_axis.xaxis[logE]))
    make_flux_map(sum_data_sky_map_smooth[logE],sum_bkgd_sky_map_smooth[logE],sum_flux_sky_map[logE],sum_flux_err_sky_map[logE],avg_energy,delta_energy)
    PlotSkyMap(fig,sum_flux_sky_map[logE],f'{source_name}_flux_sky_map_logE{logE}',roi_x=[],roi_y=[],roi_r=[])
    PlotSkyMap(fig,sum_bkgd_sky_map_smooth[logE],f'{source_name}_norm_sky_map_logE{logE}',roi_x=[],roi_y=[],roi_r=[],layer=0)
    PlotSkyMap(fig,sum_bkgd_sky_map_smooth[logE],f'{source_name}_bkgd_sky_map_logE{logE}',roi_x=[],roi_y=[],roi_r=[],layer=1)
    PlotSkyMap(fig,sum_excess_sky_map_smooth[logE],f'{source_name}_excess_sky_map_logE{logE}',roi_x=[],roi_y=[],roi_r=[],layer=0)
    sum_flux_sky_map_allE.add(sum_flux_sky_map[logE])
    sum_flux_err_sky_map_allE.addSquare(sum_flux_err_sky_map[logE])

PrintFluxCalibration(fig,sum_flux_sky_map,sum_flux_err_sky_map,roi_x,roi_y,roi_r,excl_roi_x,excl_roi_y,excl_roi_r)

for logE in range(0,logE_nbins):
    radial_axis, profile_axis, profile_err_axis = GetRadialProfile(sum_flux_sky_map[logE],sum_flux_err_sky_map[logE],roi_x[0],roi_y[0],2.0)
    baseline_yaxis = [0. for i in range(0,len(radial_axis))]
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'angular distance'
    label_y = 'surface brightness'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(radial_axis, baseline_yaxis, color='b', ls='dashed')
    axbig.errorbar(radial_axis,profile_axis,profile_err_axis,color='k',marker='s',ls='none')
    fig.savefig(f'output_plots/{source_name}_surface_brightness_logE{logE}.png',bbox_inches='tight')
    axbig.remove()
radial_axis, profile_axis, profile_err_axis = GetRadialProfile(sum_flux_sky_map_allE,sum_flux_err_sky_map_allE,roi_x[0],roi_y[0],2.0)
baseline_yaxis = [0. for i in range(0,len(radial_axis))]
fig.clf()
axbig = fig.add_subplot()
label_x = 'angular distance'
label_y = 'surface brightness'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.plot(radial_axis, baseline_yaxis, color='b', ls='dashed')
axbig.errorbar(radial_axis,profile_axis,profile_err_axis,color='k',marker='s',ls='none')
fig.savefig(f'output_plots/{source_name}_surface_brightness_allE.png',bbox_inches='tight')
axbig.remove()

for logE in range(0,logE_nbins):

    PlotSkyMap(fig,sum_significance_sky_map[logE],f'{source_name}_significance_sky_map_logE{logE}',roi_x=[],roi_y=[],roi_r=[],max_z=5.)

    max_z = 5.
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Xoff'
    label_y = 'Yoff'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    xmin = sum_init_err_xyoff_map[logE].xaxis.min()
    xmax = sum_init_err_xyoff_map[logE].xaxis.max()
    ymin = sum_init_err_xyoff_map[logE].yaxis.min()
    ymax = sum_init_err_xyoff_map[logE].yaxis.max()
    im = axbig.imshow(sum_init_err_xyoff_map[logE].waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),vmin=-max_z,vmax=max_z,aspect='auto',cmap='coolwarm')
    cbar = fig.colorbar(im)
    fig.savefig(f'output_plots/{source_name}_xyoff_init_err_map_logE{logE}.png',bbox_inches='tight')
    axbig.remove()

    for gcut in range(0,gcut_bins):
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'Xoff'
        label_y = 'Yoff'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        xmin = sum_data_xyoff_map[logE].xaxis.min()
        xmax = sum_data_xyoff_map[logE].xaxis.max()
        ymin = sum_data_xyoff_map[logE].yaxis.min()
        ymax = sum_data_xyoff_map[logE].yaxis.max()
        im = axbig.imshow(sum_data_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
        cbar = fig.colorbar(im)
        fig.savefig(f'output_plots/{source_name}_xyoff_map_logE{logE}_gcut{gcut}_data.png',bbox_inches='tight')
        axbig.remove()

        #fig.clf()
        #axbig = fig.add_subplot()
        #label_x = 'Xoff'
        #label_y = 'Yoff'
        #axbig.set_xlabel(label_x)
        #axbig.set_ylabel(label_y)
        #xmin = sum_fit_xyoff_map[logE].xaxis.min()
        #xmax = sum_fit_xyoff_map[logE].xaxis.max()
        #ymin = sum_fit_xyoff_map[logE].yaxis.min()
        #ymax = sum_fit_xyoff_map[logE].yaxis.max()
        #im = axbig.imshow(sum_fit_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
        #cbar = fig.colorbar(im)
        #fig.savefig(f'output_plots/{source_name}_xyoff_map_logE{logE}_gcut{gcut}_fit.png',bbox_inches='tight')
        #axbig.remove()

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'Xoff'
        label_y = 'Yoff'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        xmin = sum_err_xyoff_map[logE].xaxis.min()
        xmax = sum_err_xyoff_map[logE].xaxis.max()
        ymin = sum_err_xyoff_map[logE].yaxis.min()
        ymax = sum_err_xyoff_map[logE].yaxis.max()
        im = axbig.imshow(sum_err_xyoff_map[logE].waxis[:,:,gcut].T,origin='lower',extent=(xmin,xmax,ymin,ymax),vmin=-max_z,vmax=max_z,aspect='auto',cmap='coolwarm')
        cbar = fig.colorbar(im)
        fig.savefig(f'output_plots/{source_name}_xyoff_err_map_logE{logE}_gcut{gcut}.png',bbox_inches='tight')
        axbig.remove()

PlotSkyMap(fig,sum_significance_sky_map_allE,f'{source_name}_significance_sky_map_allE',roi_x=[],roi_y=[],roi_r=[],max_z=5.)
PlotSkyMap(fig,sum_excess_sky_map_allE,f'{source_name}_excess_sky_map_allE',roi_x=[],roi_y=[],roi_r=[])

print (f'total_exposure = {total_exposure}')

for logE in range(0,logE_nbins):

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'CR chi2'
    label_y = 'SR chi2'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    for entry in range(0,len(list_sr_qual)):
        axbig.scatter(list_cr_qual[entry][logE],list_sr_qual[entry][logE],color='b',alpha=0.5)
    #axbig.set_xscale('log')
    #axbig.set_yscale('log')
    #axbig.set_xlim(0.1,10.)
    fig.savefig(f'output_plots/{source_name}_crsr_qual_logE{logE}.png',bbox_inches='tight')
    axbig.remove()
    
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Run elevation'
    label_y = 'SR chi2'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    for entry in range(0,len(list_sr_qual)):
        axbig.scatter(list_run_elev[entry],list_sr_qual[entry][logE],color='b',alpha=0.5)
    fig.savefig(f'output_plots/{source_name}_elev_sr_qual_logE{logE}.png',bbox_inches='tight')
    axbig.remove()
    
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Run azimuth'
    label_y = 'SR chi2'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    for entry in range(0,len(list_sr_qual)):
        axbig.scatter(list_run_azim[entry],list_sr_qual[entry][logE],color='b',alpha=0.5)
    fig.savefig(f'output_plots/{source_name}_azim_sr_qual_logE{logE}.png',bbox_inches='tight')
    axbig.remove()

fig.clf()
axbig = fig.add_subplot()
label_x = 'elevation'
label_y = 'azimuth'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
for entry in range(0,len(list_run_elev)):
    axbig.scatter(list_run_elev[entry],list_run_azim[entry],color='b',alpha=0.5)
fig.savefig(f'output_plots/{source_name}_elevazim.png',bbox_inches='tight')
axbig.remove()

fig.clf()
axbig = fig.add_subplot()
label_x = 'elevation'
axbig.set_xlabel(label_x)
axbig.hist(list_run_elev, bins=20)
fig.savefig(f'output_plots/{source_name}_elev.png',bbox_inches='tight')
axbig.remove()

#for par1 in range(0,matrix_rank):
#    for par2 in range(par1+1,matrix_rank):
#        fig.clf()
#        axbig = fig.add_subplot()
#        label_x = 'c%s'%(par1)
#        label_y = 'c%s'%(par2)
#        axbig.set_xlabel(label_x)
#        axbig.set_ylabel(label_y)
#        for entry in range(0,len(list_truth_params)):
#            axbig.scatter(list_truth_params[entry][par1],list_truth_params[entry][par2],color='b',alpha=0.5)
#            axbig.scatter(list_fit_params[entry][par1],list_fit_params[entry][par2],color='r',alpha=0.5)
#        fig.savefig(f'output_plots/{source_name}_truth_params_c{par1}_c{par2}.png',bbox_inches='tight')
#        axbig.remove()

for par1 in range(0,len(list_truth_params[0])):

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'truth'
    label_y = 'fit'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    for entry in range(0,len(list_truth_params)):
        axbig.scatter(list_truth_params[entry][par1],list_fit_params[entry][par1],color='b',alpha=0.5)
    fig.savefig(f'output_plots/{source_name}_truth_fit_params_c{par1}.png',bbox_inches='tight')
    axbig.remove()

