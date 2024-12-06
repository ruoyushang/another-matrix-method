# another-matrix-method

Here are the steps for the method:
1. Search for ON runs and matched OFF runs in the database
2. Prepare the ED/VEGAS ROOT files for both ON and OFF runs
3. Serialize the background maps of the OFF runs and build a large matrix
4. SVD the large metrix and build eigenvectors of the background templates
5. Analyze the ON runs and use the eigenvectors to predict background model
6. Make plots

Before runnign the code, you need to define these env variables:
export SMI_INPUT=/THIS/IS/WHERE/YOUR/ROOT/FILES/ARE
export SMI_AUX=/DIRECTORY/FOR/DATABASE/AUX/FILES
export SMI_DIR=/THIS/IS/WHERE/YOUR/WORKING/DIR
export SMI_RUNLIST=/THIS/STORES/DATABASE/JOB/OUTPUTS
export SMI_OUTPUT=/THIS/STORES/YOUR/ANALYSIS/OUTPUTS
export SKY_TAG="fullspec16"

You will need to create a "run" folder and a "output_plots" folder in your working directory.

1. Search for ON runs and matched OFF runs in the database:
The script you will be using is veritas_db_query.py, 
you will need to define an output dir in the script, 
e.g. output_dir = "output_vts_query_default", and create this dir in your working space.
Then you can run "python3 make_vts_db_script.py", which will create shell scripts in the "run" folder, e.g. "vts_db_Geminga.sh"
You can execute "sh vts_db_Geminga.sh", and the code will search for ON runs and OFF runs in the database and save the run lists in the "output_vts_query_default" folder.
You can add new sources in make_vts_db_script.py, e.g.
input_params += [ ['source_name', source_RA, source_DEC  , min_elevation, max_elevation] ]
 
2. Prepare the ED/VEGAS ROOT files for both ON and OFF runs:
Use Geminga as an example, you can find the ON runlist in this file: output_vts_query_default/RunList_Geminga_V6.txt
and the OFF runs in output_vts_query_default/PairList_Geminga_V6.txt 
In PairList_Geminga_V6.txt, the first column is the ON runs, and the second column is the matched OFF runs.


3. Serialize the background maps of the OFF runs and build a large matrix:
Run "python3 make_condor_scripts.py" to create shell scripts for the matrix method jobs. The scripts will be stored in the "run" folder.
Execute "sh run/save_mtx_Geminga_ON.sh", which will read all OFF-run ROOT files and build a large matrix of the OFF-run background templates.
The output files will be stored in $SMI_OUTPUT

4. SVD the large metrix and build eigenvectors of the background templates:
Execute "sh run/eigenvtr_Geminga_ON.sh", which will perform singular value decomposition on the large matrix and build eigenvectors of the background template.
The output files will be stored in $SMI_OUTPUT, and you can see plots of the eigenvectors in "output_plots" folder.

5. Analyze the ON runs and use the eigenvectors to predict background model:
Execute "sh run/skymap_Geminga_ON.sh", which will analyze the ON runs, produce sky maps, and store output files in $SMI_OUTPUT.

6. Make plots:
Execute "sh run/plot_Geminga_ON.sh", and your analysis plots will be created in the "output_plots" folder.

