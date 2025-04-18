
#module load root/06.24.00

conda init
conda deactivate
conda activate /nevis/ged/data/rshang/my_conda_envs/root_env

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

export SMI_INPUT=/nevis/ged/data/rshang/EDv490_output
#export SMI_INPUT=/nevis/ged/data/rshang/EDv490_output_PSR_J2021_p4026
#export SMI_INPUT=/nevis/ged/data/rshang/EDv490_output_HESS_J0632_p057
echo $SMI_INPUT


export SMI_AUX=/nevis/ged/data/rshang/SMI_AUX
export SMI_DIR=/nevis/tehanu/home/ryshang/veritas_analysis/another-matrix-method

#export SMI_RUNLIST=/nevis/tehanu/home/ryshang/veritas_analysis/another-matrix-method/output_vts_query
#export SMI_OUTPUT=/nevis/ged/data/rshang/smi_output/output_test_3
#export SMI_OUTPUT=/nevis/ged/data/rshang/smi_output/output_test_4

#export SMI_RUNLIST=/nevis/tehanu/home/ryshang/veritas_analysis/another-matrix-method/output_vts_query_20241228
#export SMI_OUTPUT=/nevis/ged/data/rshang/smi_output/output_test
#export SMI_RUNLIST=/nevis/tehanu/home/ryshang/veritas_analysis/another-matrix-method/output_vts_query_20241228
#export SMI_OUTPUT=/nevis/ged/data/rshang/smi_output/output_default
export SMI_RUNLIST=/nevis/tehanu/home/ryshang/veritas_analysis/another-matrix-method/output_vts_query_20241228
export SMI_OUTPUT=/nevis/ged/data/rshang/smi_output/output_20250417
#export SMI_RUNLIST=/nevis/tehanu/home/ryshang/veritas_analysis/another-matrix-method/output_vts_query_20241228
#export SMI_OUTPUT=/nevis/ged/data/rshang/smi_output/output_hnsb
#export SMI_RUNLIST=/nevis/tehanu/home/ryshang/veritas_analysis/another-matrix-method/output_vts_query_20241228
#export SMI_OUTPUT=/nevis/ged/data/rshang/smi_output/output_lnsb

echo "SMI_RUNLIST="$SMI_RUNLIST
echo "SMI_OUTPUT="$SMI_OUTPUT

#export CR_TAG="cr6"
export CR_TAG="cr8"

#export NORM_TAG="fov03"
#export NORM_TAG="fov05"
#export NORM_TAG="fov10"
#export NORM_TAG="fov15"
export NORM_TAG="free"

export BIN_TAG="nbin0"
#export BIN_TAG="nbin1"
#export BIN_TAG="nbin3"
#export BIN_TAG="nbin5"
#export BIN_TAG="nbin7"
#export BIN_TAG="nbin9"
#export BIN_TAG="nbin10"

#export EIGEN_TAG="init"
#export EIGEN_TAG="fullspec1"
#export EIGEN_TAG="fullspec2"
#export EIGEN_TAG="fullspec4"
#export EIGEN_TAG="fullspec8"
#export EIGEN_TAG="fullspec16"
#export EIGEN_TAG="fullspec32"
export EIGEN_TAG="fullspec64"
#export EIGEN_TAG="fullspec128"
#export EIGEN_TAG="monospec8"

export SKY_TAG=$CR_TAG"_"$BIN_TAG"_"$EIGEN_TAG"_"$NORM_TAG

export ANA_DIR=$CR_TAG"_"$BIN_TAG
export MTX_DIR=$SMI_OUTPUT/$CR_TAG"_"$BIN_TAG

echo $SKY_TAG

