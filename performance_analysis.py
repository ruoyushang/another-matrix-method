
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

#ana_tag = 'nominal'
#ana_tag = 'poisson'
ana_tag = '1em2'

onoff = 'OFF'

exposure_per_group = 0.2
#exposure_per_group = 1.
#exposure_per_group = 5.
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
input_sources += [ ['CrabNebula_elev_80_90' ,83.633  ,22.014 ] ]
input_sources += [ ['CrabNebula_elev_70_80' ,83.633  ,22.014 ] ]
input_sources += [ ['CrabNebula_elev_60_70' ,83.633  ,22.014 ] ]
input_sources += [ ['CrabNebula_elev_50_60' ,83.633  ,22.014 ] ]
input_sources += [ ['CrabNebula_elev_40_50' ,83.633  ,22.014 ] ]
input_sources += [ ['CrabNebula_elev_30_40' ,83.633  ,22.014 ] ]
input_sources += [ ['CrabNebula_1p0wobble' ,83.633  ,22.014 ] ]
input_sources += [ ['CrabNebula_1p5wobble' ,83.633  ,22.014 ] ]

data_count = []
bkgd_count = []
for logE in range(0,logE_nbins):
    data_count += [0.]
    bkgd_count += [0.]

grp_avg_elev = []
grp_data_count = []
grp_bkgd_count = []

list_sr_qual = []
list_cr_qual = []
list_truth_params = []
list_fit_params = []

total_exposure = 0.
good_exposure = 0.
group_exposure = 0.
avg_elev = 0.
for epoch in input_epoch:
    for src in input_sources:

        source_name = src[0]

        input_filename = f'{smi_output}/skymaps_{source_name}_{epoch}_{onoff}_{ana_tag}.pkl'
        #print (f'reading {input_filename}...')
        if not os.path.exists(input_filename):
            #print (f'{input_filename} does not exist.')
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
            data_sky_map = analysis_result[run][2] 
            bkgd_sky_map = analysis_result[run][3] 

            if run_azim>270.:
                run_azim = run_azim-360.

            if run_elev<min_elev:
                continue

            total_exposure += exposure

            bkgd_sum = 0.
            for logE in range(0,logE_nbins):
                bkgd_sum += np.sum(bkgd_sky_map[logE].waxis[:,:,:])

            is_good_run = True
            if cr_qual>cr_qual_cut:
                is_good_run = False
            if not is_good_run: 
                print (f'bad fitting. reject the run.')
                continue

            list_sr_qual += [sr_qual]
            list_cr_qual += [cr_qual]
            list_truth_params += [truth_params]
            list_fit_params += [fit_params]

            good_exposure += exposure
            group_exposure += exposure
            avg_elev += exposure*run_elev

            for logE in range(0,logE_nbins):
            
                data_sum = np.sum(data_sky_map[logE].waxis[:,:,0])
                bkgd_sum = np.sum(bkgd_sky_map[logE].waxis[:,:,0])
                norm_sum = pow(data_sum*data_sum+bkgd_sum*bkgd_sum,0.5)
                if norm_sum>0.:
                    significance = (data_sum-bkgd_sum)/pow(norm_sum,0.5)
                    if significance>10.:
                        print (f'large error in input_filename = {input_filename}')
                data_count[logE] += data_sum
                bkgd_count[logE] += bkgd_sum

            if group_exposure>exposure_per_group or run==len(analysis_result)-1:

                if group_exposure>0.5*exposure_per_group:
                    tmp_data_count = []
                    tmp_bkgd_count = []
                    for logE in range(0,logE_nbins):
                        tmp_data_count += [data_count[logE]]
                        tmp_bkgd_count += [bkgd_count[logE]]
                    avg_elev = avg_elev/group_exposure
                    grp_avg_elev += [avg_elev]
                    grp_data_count += [tmp_data_count]
                    grp_bkgd_count += [tmp_bkgd_count]

                #print ('=================================================================================')
                #for logE in range(0,logE_nbins):
                #    error = 0.
                #    stat_error = 0.
                #    if data_count[logE]>0.:
                #        error = 100.*(data_count[logE]-bkgd_count[logE])/data_count[logE]
                #        stat_error = 100.*pow(data_count[logE],0.5)/data_count[logE]
                #    print (f'E = {pow(10.,logE_bins[logE]):0.3f} TeV, data_sum = {data_count[logE]}, bkgd_sum = {bkgd_count[logE]:0.1f}, error = {error:0.1f} +/- {stat_error:0.1f} %')

                group_exposure = 0.
                avg_elev = 0.
                for logE in range(0,logE_nbins):
                    data_count[logE] = 0.
                    bkgd_count[logE] = 0.

print (f'total_exposure = {total_exposure:0.1f}, good_exposure = {good_exposure:0.1f}')
print (f'ana_tag = {ana_tag}, len(grp_data_count) = {len(grp_data_count)}, exposure_per_group = {exposure_per_group}')
print ('=================================================================================')
bias_array = []
for logE in range(0,logE_nbins):
    avg_data = 0.
    avg_bkgd = 0.
    sum_weight = 0.
    for grp in range(0,len(grp_data_count)):
        data = grp_data_count[grp][logE]
        bkgd = grp_bkgd_count[grp][logE]
        weight = data
        if data>0.:
            avg_data += data*weight
            avg_bkgd += bkgd*weight
            sum_weight += weight
    avg_data = avg_data/sum_weight
    avg_bkgd = avg_bkgd/sum_weight
    avg_bias = 100.*(avg_data-avg_bkgd)/avg_data
    bias_array += [avg_bias/100.]
    avg_error = 0.
    avg_stat_error = 0.
    sum_weight = 0.
    for grp in range(0,len(grp_data_count)):
        data = grp_data_count[grp][logE]
        bkgd = grp_bkgd_count[grp][logE]
        #bkgd = grp_bkgd_count[grp][logE] * (1.+avg_bias/100.)
        weight = data
        if data>0.:
            #avg_error += pow((data-bkgd)/data,2)*weight
            #avg_stat_error += 2.*1./data*weight
            avg_error += abs(data-bkgd)/data*weight
            avg_stat_error += pow(data,0.5)/data*weight
            sum_weight += weight
    #avg_error = 100.*pow(avg_error/sum_weight,0.5)
    #avg_stat_error = 100.*pow(avg_stat_error/sum_weight,0.5)
    avg_error = 100.*avg_error/sum_weight
    avg_stat_error = 100.*avg_stat_error/sum_weight
    print (f'E = {pow(10.,logE_bins[logE]):0.3f} TeV, avg_data = {avg_data:0.1f}, avg_bkgd = {avg_bkgd:0.1f}, bias = {avg_bias:0.1f} %, error = {avg_error:0.1f} +/- {avg_stat_error:0.1f} %')

print (f'bias_array = {np.around(bias_array,3)}')

for logE in range(0,logE_nbins):
    list_error_significance = []
    list_elev = []
    for grp in range(0,len(grp_data_count)):
        elev = grp_avg_elev[grp]
        data = grp_data_count[grp][logE]
        bkgd = grp_bkgd_count[grp][logE]
        if data==0.: continue
        list_elev += [elev]
        list_error_significance += [(data-bkgd)/pow(data,0.5)]
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Elevation [deg]'
    label_y = 'Error significance'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_elev,list_error_significance,color='b',alpha=0.1)
    fig.savefig(f'output_plots/error_vs_elev_logE{logE}_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()


hist_range = [[0.,  1.], [-5., 5.]]

fig.clf()
axbig = fig.add_subplot()
label_x = 'CR chi2'
label_y = 'SR chi2'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.scatter(list_cr_qual,list_sr_qual,color='b',alpha=0.5)
#axbig.hist2d(list_cr_qual,list_sr_qual, norm=mpl.colors.LogNorm(), bins=50, range=hist_range)
axbig.set_xscale('log')
#axbig.set_yscale('log')
#axbig.set_ylim(-5.,5.)
fig.savefig(f'output_plots/all_src_crsr_qual_{ana_tag}.png',bbox_inches='tight')
axbig.remove()

plot_n_params = 3
plot_truth_params = []
plot_fit_params = []
for par in range(0,plot_n_params):
    plot_truth_params += [None]
    plot_fit_params += [None]

for entry in range(0,len(list_truth_params)):
    for par in range(0,len(list_truth_params[entry])):
        if par>=plot_n_params: continue
        truth = list_truth_params[entry][par]
        fit = list_fit_params[entry][par]
        if truth==0.: continue
        if plot_truth_params[par]==None:
            plot_truth_params[par] = [truth]
            plot_fit_params[par] = [fit]
        else:
            plot_truth_params[par] += [truth]
            plot_fit_params[par] += [fit]


#for par in range(0,plot_n_params):
#    fig.clf()
#    axbig = fig.add_subplot()
#    label_x = 'fit'
#    label_y = 'error'
#    axbig.set_xlabel(label_x)
#    axbig.set_ylabel(label_y)
#    truth = np.array(plot_truth_params[par])
#    fit = np.array(plot_fit_params[par])
#    axbig.scatter(abs(fit),(fit-truth)/pow(abs(truth),0.5),color='b',alpha=0.5)
#    fig.savefig(f'output_plots/truth_fit_params_c{par}.png',bbox_inches='tight')
#    axbig.remove()

exit()
    
