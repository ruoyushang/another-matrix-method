
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

fig, ax = plt.subplots()
figsize_x = 6.4
figsize_y = 4.6
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

smi_input = os.environ.get("SMI_INPUT")
smi_output = os.environ.get("SMI_OUTPUT")
smi_dir = os.environ.get("SMI_DIR")

#ana_tag = 'nominal'
#ana_tag = 'poisson'

ana_tag = []
#ana_tag += [['binspec','r']]
#ana_tag += [['fullspec','b']]
ana_tag += [['init','r']]
ana_tag += [['rank1','w']]
ana_tag += [['rank2','w']]
ana_tag += [['rank3','w']]
ana_tag += [['rank4','w']]
ana_tag += [['rank5','b']]
ana_tag += [['rank6','w']]
ana_tag += [['rank7','w']]
ana_tag += [['rank8','w']]

onoff = 'OFF'

#exposure_per_group = 1.
#exposure_per_group = 5.
#exposure_per_group = 10.
#exposure_per_group = 50.
exposure_per_group = 100.
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

ana_avg_elev = []
ana_data_count = []
ana_bkgd_count = []

ana_chi2_sr = []
ana_chi2_cr = []

ana_syst_err_vs_logE = []
ana_stat_err_vs_logE = []

for ana in range(0,len(ana_tag)):

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

    list_chi2_sr = []
    list_chi2_cr = []
    
    total_exposure = 0.
    good_exposure = 0.
    group_exposure = 0.
    avg_elev = 0.

    for epoch in input_epoch:
        for src in input_sources:
    
            source_name = src[0]
    
            input_filename = f'{smi_output}/skymaps_{source_name}_{epoch}_{onoff}_{ana_tag[ana][0]}.pkl'
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
                data_xyoff_map = analysis_result[run][4]
                bkgd_xyoff_map = analysis_result[run][5]
    
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

                logE_peak = 0
                bkgd_peak = 0.
                for logE in range(0,logE_nbins):
                    bkgd = np.sum(bkgd_xyoff_map[logE].waxis[:,:,:])
                    if bkgd>bkgd_peak:
                        bkgd_peak = bkgd
                        logE_peak = logE
    
                for logE in range(0,logE_nbins):

                    if logE<logE_peak: continue
                
                    data_sum = np.sum(data_sky_map[logE].waxis[:,:,0])
                    bkgd_sum = np.sum(bkgd_sky_map[logE].waxis[:,:,0])
                    norm_sum = pow(data_sum*data_sum+bkgd_sum*bkgd_sum,0.5)
                    #if norm_sum>0.:
                    #    significance = (data_sum-bkgd_sum)/pow(norm_sum,0.5)
                    #    if abs(significance)>10.:
                    #        print (f'large error in input_filename = {input_filename}')
                    data_count[logE] += data_sum
                    bkgd_count[logE] += bkgd_sum

                    for binx in range(0,len(data_xyoff_map[logE].xaxis)-1):
                        for biny in range(0,len(data_xyoff_map[logE].yaxis)-1):

                            data = data_xyoff_map[logE].waxis[binx,biny,0]
                            bkgd = bkgd_xyoff_map[logE].waxis[binx,biny,0]
                            if data<10: continue
                            chi2 = (data-bkgd)/pow(data,0.5)
                            list_chi2_sr += [chi2]

                            for cr in range(1,4):
                                data = data_xyoff_map[logE].waxis[binx,biny,cr]
                                bkgd = bkgd_xyoff_map[logE].waxis[binx,biny,cr]
                                if data<10: continue
                                chi2 = (data-bkgd)/pow(data,0.5)
                                list_chi2_cr += [chi2]

    
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

                    data_sum = 0.
                    bkgd_sum = 0.
                    for logE in range(0,logE_nbins):
                        data_sum += data_count[logE]
                        bkgd_sum += bkgd_count[logE]
                    norm_sum += pow(data_sum*data_sum+bkgd_sum*bkgd_sum,0.5)
                    if norm_sum>0.:
                        significance = (data_sum-bkgd_sum)/pow(norm_sum,0.5)
                        #if abs(significance)>10.:
                        #    print (f'large error in input_filename = {input_filename}, significance = {significance:0.2f}')
    
                    group_exposure = 0.
                    avg_elev = 0.
                    for logE in range(0,logE_nbins):
                        data_count[logE] = 0.
                        bkgd_count[logE] = 0.
    
    print (f'total_exposure = {total_exposure:0.1f}, good_exposure = {good_exposure:0.1f}')
    print (f'ana_tag = {ana_tag[ana][0]}, len(grp_data_count) = {len(grp_data_count)}, exposure_per_group = {exposure_per_group}')
    print ('=================================================================================')
    bias_array = []
    syst_err_vs_logE_array = []
    stat_err_vs_logE_array = []
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
        if sum_weight==0.: continue
        avg_data = avg_data/sum_weight
        avg_bkgd = avg_bkgd/sum_weight
        avg_bias = 100.*(avg_data-avg_bkgd)/avg_data
        bias_array += [avg_bias/100.]
        avg_error = 0.
        avg_error_correct = 0.
        avg_stat_error = 0.
        sum_weight = 0.
        for grp in range(0,len(grp_data_count)):
            data = grp_data_count[grp][logE]
            bkgd = grp_bkgd_count[grp][logE]
            bkgd_correct = grp_bkgd_count[grp][logE] * (1.+avg_bias/100.)
            weight = data
            if data>0.:
                #avg_error += pow((data-bkgd)/data,2)*weight
                #avg_stat_error += 2.*1./data*weight
                avg_error += abs(data-bkgd)/data*weight
                avg_error_correct += abs(data-bkgd_correct)/data*weight
                avg_stat_error += pow(data,0.5)/data*weight
                sum_weight += weight
        #avg_error = 100.*pow(avg_error/sum_weight,0.5)
        #avg_stat_error = 100.*pow(avg_stat_error/sum_weight,0.5)
        avg_error = 100.*avg_error/sum_weight
        avg_error_correct = 100.*avg_error_correct/sum_weight
        avg_stat_error = 100.*avg_stat_error/sum_weight
        print (f'E = {pow(10.,logE_bins[logE]):0.3f} TeV, avg_data = {avg_data:0.1f}, avg_bkgd = {avg_bkgd:0.1f}, bias = {avg_bias:0.2f} %, error = {avg_error:0.2f} +/- {avg_stat_error:0.2f} %, error (correct) = {avg_error_correct:0.2f} +/- {avg_stat_error:0.2f} %')

        syst_err_vs_logE_array += [avg_error]
        stat_err_vs_logE_array += [avg_stat_error]
    
    print (f'bias_array = {np.around(bias_array,4)}')

    ana_avg_elev += [grp_avg_elev]
    ana_data_count += [grp_data_count]
    ana_bkgd_count += [grp_bkgd_count]

    ana_chi2_sr += [list_chi2_sr]
    ana_chi2_cr += [list_chi2_cr]

    ana_syst_err_vs_logE += [syst_err_vs_logE_array]
    ana_stat_err_vs_logE += [stat_err_vs_logE_array]

energy_axis = []
for logE in range(0,logE_nbins):
    energy_axis += [pow(10.,logE_bins[logE])]

for logE in range(0,logE_nbins):

    ana_axis = []
    for ana in range(0,len(ana_tag)):
        ana_axis += [ana]

    syst_err_axis = []
    stat_err_axis = []
    for ana in range(0,len(ana_tag)):
        syst_err_axis += [ana_syst_err_vs_logE[ana][logE]]
        stat_err_axis += [ana_stat_err_vs_logE[ana][logE]]

    fig.clf()
    axbig = fig.add_subplot()
    label_x = '$k$'
    label_y = '$\epsilon$ (%)'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    E_min = pow(10.,logE_bins[logE])
    E_max = pow(10.,logE_bins[logE+1])
    axbig.errorbar(ana_axis,syst_err_axis,yerr=stat_err_axis,color='k',marker='.',ls='solid',label=f'E = {E_min:0.2f} - {E_max:0.2f} TeV')
    axbig.legend(loc='best')
    fig.savefig(f'output_plots/ana_syst_err_logE{logE}.png',bbox_inches='tight')
    axbig.remove()


ratio_cut = 0.3
for logE in range(0,logE_nbins):
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Elevation [deg]'
    label_y = 'Error significance'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)

    for ana in range(0,len(ana_tag)):
        if ana_tag[ana][1]=='w': continue
        list_error_significance = []
        list_elev = []
        for grp in range(0,len(ana_avg_elev[ana])):
            elev = ana_avg_elev[ana][grp]
            data = ana_data_count[ana][grp][logE]
            bkgd = ana_bkgd_count[ana][grp][logE]
            if data==0.: continue
            error_ratio = (data-bkgd)/(data)
            if abs(error_ratio)>ratio_cut: continue
            list_elev += [elev]
            list_error_significance += [(data-bkgd)/pow(data,0.5)]
        axbig.scatter(list_elev,list_error_significance,color=ana_tag[ana][1],alpha=0.3)

    fig.savefig(f'output_plots/error_significance_vs_elev_logE{logE}.png',bbox_inches='tight')
    axbig.remove()

for logE in range(0,logE_nbins):
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Elevation [deg]'
    label_y = 'Relative error'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)

    for ana in range(0,len(ana_tag)):
        if ana_tag[ana][1]=='w': continue
        list_error_significance = []
        list_elev = []
        for grp in range(0,len(ana_avg_elev[ana])):
            elev = ana_avg_elev[ana][grp]
            data = ana_data_count[ana][grp][logE]
            bkgd = ana_bkgd_count[ana][grp][logE]
            if data==0.: continue
            error_ratio = (data-bkgd)/(data)
            if abs(error_ratio)>ratio_cut: continue
            list_elev += [elev]
            list_error_significance += [(data-bkgd)/(data)]
        axbig.scatter(list_elev,list_error_significance,color=ana_tag[ana][1],alpha=0.3)

    fig.savefig(f'output_plots/error_ratio_vs_elev_logE{logE}.png',bbox_inches='tight')
    axbig.remove()

for ana in range(0,len(ana_tag)):

    hist_binsize = 0.1
    chi2_axis = np.arange(-5., 5., 0.01)
    total_entries = len(ana_chi2_sr[ana])
    normal_dist = hist_binsize*total_entries/pow(2.*np.pi,0.5)*np.exp(-chi2_axis*chi2_axis/2.)
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'significance'
    label_y = 'number of entries'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    hist_chi2, bin_edges = np.histogram(ana_chi2_sr[ana],bins=100,range=(-5.,5.))
    axbig.hist(ana_chi2_sr[ana], bin_edges)
    axbig.plot(chi2_axis,normal_dist)
    #axbig.set_yscale('log')
    fig.savefig(f'output_plots/chi2_sr_distribution_{ana_tag[ana][0]}.png',bbox_inches='tight')
    axbig.remove()
    
    chi2_axis = np.arange(-5., 5., 0.01)
    total_entries = len(ana_chi2_cr[ana])
    normal_dist = hist_binsize*total_entries/pow(2.*np.pi,0.5)*np.exp(-chi2_axis*chi2_axis/2.)
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'significance'
    label_y = 'number of entries'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    hist_chi2, bin_edges = np.histogram(ana_chi2_cr[ana],bins=100,range=(-5.,5.))
    axbig.hist(ana_chi2_cr[ana], bin_edges)
    axbig.plot(chi2_axis,normal_dist)
    #axbig.set_yscale('log')
    fig.savefig(f'output_plots/chi2_cr_distribution_{ana_tag[ana][0]}.png',bbox_inches='tight')
    axbig.remove()

#hist_range = [[0.,  1.], [-5., 5.]]
#
#fig.clf()
#axbig = fig.add_subplot()
#label_x = 'CR chi2'
#label_y = 'SR chi2'
#axbig.set_xlabel(label_x)
#axbig.set_ylabel(label_y)
#axbig.scatter(list_cr_qual,list_sr_qual,color='b',alpha=0.5)
##axbig.hist2d(list_cr_qual,list_sr_qual, norm=mpl.colors.LogNorm(), bins=50, range=hist_range)
#axbig.set_xscale('log')
##axbig.set_yscale('log')
##axbig.set_ylim(-5.,5.)
#fig.savefig(f'output_plots/all_src_crsr_qual_{ana_tag}.png',bbox_inches='tight')
#axbig.remove()

#plot_n_params = 3
#plot_truth_params = []
#plot_fit_params = []
#for par in range(0,plot_n_params):
#    plot_truth_params += [None]
#    plot_fit_params += [None]
#
#for entry in range(0,len(list_truth_params)):
#    for par in range(0,len(list_truth_params[entry])):
#        if par>=plot_n_params: continue
#        truth = list_truth_params[entry][par]
#        fit = list_fit_params[entry][par]
#        if truth==0.: continue
#        if plot_truth_params[par]==None:
#            plot_truth_params[par] = [truth]
#            plot_fit_params[par] = [fit]
#        else:
#            plot_truth_params[par] += [truth]
#            plot_fit_params[par] += [fit]

#plot_n_params = 2
#fig.clf()
#figsize_x = 2.*plot_n_params
#figsize_y = 2.*plot_n_params
#fig.set_figheight(figsize_y)
#fig.set_figwidth(figsize_x)
#for par1 in range(0,plot_n_params):
#    for par2 in range(par1,plot_n_params):
#        truth_param1 = []
#        truth_param2 = []
#        for run in range(0,len(list_truth_params)):
#            if par1>=len(list_truth_params[run]): continue
#            ax_idx = par2 + par1*plot_n_params + 1
#            axbig = fig.add_subplot(plot_n_params,plot_n_params,ax_idx)
#            if par1==0:
#                axbig.set_title(f'Par {par2}')
#            if par2==0:
#                axbig.set_ylabel(f'Par {par1}')
#            truth_param1 += [list_truth_params[run][par1]]
#            truth_param2 += [list_truth_params[run][par2]]
#        axbig.scatter(truth_param1,truth_param2,color='b',alpha=0.5)
#fig.savefig(f'output_plots/param_cov_matrix.png',bbox_inches='tight')
#axbig.remove()

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
    
