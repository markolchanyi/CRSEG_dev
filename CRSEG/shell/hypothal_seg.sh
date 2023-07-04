export FREESURFER_HOME=/usr/local/freesurfer/7.4.1
source $FREESURFER_HOME/SetUpFreeSurfer.sh
source /usr/pubsw/packages/mrtrix/env.sh
mrtrixdir=/usr/pubsw/packages/mrtrix/3.0.2/bin
fsldir=/usr/pubsw/packages/fsl/current/bin
FSLDIR=/usr/pubsw/packages/fsl/current
. ${FSLDIR}/etc/fslconf/fsl.sh
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH

export LD_LIBRARY_PATH=/usr/pubsw/packages/CUDA/9.1/lib64

mri_segment_hypothalamic_subunits --i $1 --o $2 --cpu
