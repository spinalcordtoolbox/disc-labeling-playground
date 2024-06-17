import os
import subprocess
import shutil
from ply.utils.utils import cropWithSC

def fetch_and_preproc_image_cGAN_withSeg(path_in, path_seg, tmpdir):
    '''
    :param path_in: Path to the input image
    :param path_seg: Path to the input SC segmentation
    :param tmpdir: Path to tempdirectory
    :return: out_decathlon_monai: list of dictionary with the preprocessed image path (like monai load_decathlon_datalist)
        [
            {'image': '/workspace/data/chest_19.nii.gz',
        ]
    '''
    # Check if paths exist
    path_in = os.path.abspath(path_in)
    path_seg = os.path.abspath(path_seg)
    if not os.path.exists(path_in) or not os.path.exists(path_seg):
        raise ValueError(f'Error with path: either {path_in} or {path_seg} does not exist')
    
    # Create temp_in and temp_seg
    temp_in_path = os.path.join(tmpdir, os.path.basename(path_in))
    temp_seg_path = os.path.join(tmpdir, os.path.basename(path_seg))
    shutil.copyfile(path_in, temp_in_path)
    shutil.copyfile(path_seg, temp_seg_path)

    # Reorient images to RSP using SCT
    print('Reorienting images to RSP...')
    subprocess.check_call(['sct_image',
                        '-i', temp_in_path,
                        '-setorient', 'RSP'])
    
    subprocess.check_call(['sct_image',
                        '-i', temp_seg_path,
                        '-setorient', 'RSP'])
    
    # Crop input image
    temp_in_crop_path = cropWithSC(temp_in_path, temp_seg_path, tmpdir)

    return [{'image': temp_in_crop_path}]


def fetch_and_preproc_image_cGAN_NoSeg(path_in, tmpdir):
    '''
    :param path_in: Path to the input image
    :param tmpdir: Path to tempdirectory
    :return: out_decathlon_monai: list of dictionary with the preprocessed image path (like monai load_decathlon_datalist)
        [
            {'image': '/workspace/data/chest_19.nii.gz',
        ]
    '''
    # Check if paths exist
    path_in = os.path.abspath(path_in)
    if not os.path.exists(path_in):
        raise ValueError(f'Error with path: {path_in} does not exist')
    
    # Create temp_in and temp_seg
    temp_in_path = os.path.join(tmpdir, os.path.basename(path_in))
    shutil.copyfile(path_in, temp_in_path)

    # Reorient images to RSP using SCT
    print('Reorienting images to RSP...')
    subprocess.check_call(['sct_image',
                        '-i', temp_in_path,
                        '-setorient', 'RSP'])
    return [{'image': temp_in_path}]