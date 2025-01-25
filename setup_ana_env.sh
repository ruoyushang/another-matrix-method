
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
export SMI_OUTPUT=/nevis/ged/data/rshang/smi_output/output_multivar

echo "SMI_RUNLIST="$SMI_RUNLIST
echo "SMI_OUTPUT="$SMI_OUTPUT

#export CR_TAG="cr15"
export CR_TAG="cr20"
#export CR_TAG="cr25"
#export CR_TAG="cr30"
#export CR_TAG="wr05"

#export NORM_TAG="fov10"
#export NORM_TAG="fov15"
#export NORM_TAG="free"
export NORM_TAG="constraint"

#export BIN_TAG="nbin1"
#export BIN_TAG="nbin3"
#export BIN_TAG="nbin5"
export BIN_TAG="nbin7"
#export BIN_TAG="nbin9"

#export EIGEN_TAG="init"
#export EIGEN_TAG="fullspec1"
#export EIGEN_TAG="fullspec2"
#export EIGEN_TAG="fullspec3"
#export EIGEN_TAG="fullspec4"
#export EIGEN_TAG="fullspec8"
export EIGEN_TAG="fullspec16"
#export EIGEN_TAG="fullspec32"
#export EIGEN_TAG="fullspec64"
#export EIGEN_TAG="monospec8"

export SKY_TAG=$CR_TAG"_"$BIN_TAG"_"$EIGEN_TAG"_"$NORM_TAG

export ANA_DIR=$CR_TAG"_"$BIN_TAG
export MTX_DIR=$SMI_OUTPUT/$CR_TAG"_"$BIN_TAG

echo $SKY_TAG

