import nibabel as nib 
import dipy
from dipy.io.streamline import load_tractogram, save_tractogram
import os

#nib.streamlines.convert('my_trk.trk', 'my_tck.tck')
scratch_dir = ""
reference_anatomy = 0
thalamus_sft = load_tractogram(os.path.join(scratch_dir,"tracts_thal.tck"), reference_anatomy)
