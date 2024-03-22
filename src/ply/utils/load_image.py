import os
import subprocess
import shutil
from ply.utils.utils import cropWithSC

def fetch_and_preproc_image_cGAN(path_in, path_seg, tmpdir):
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
    subprocess.check_call(['sct_image',
                        '-i', temp_in_path,
                        '-setorient', 'RSP'])
    
    subprocess.check_call(['sct_image',
                        '-i', temp_seg_path,
                        '-setorient', 'RSP'])
    
    # Crop input image
    temp_in_crop_path = cropWithSC(temp_in_path, temp_seg_path, tmpdir)

    return [{'image':{temp_in_crop_path}}]

    
    

    for path in paths:
        if 'DATASETS_PATH' in config_data.keys():
            dest_sc_seg_path = os.path.join(config_data['DATASETS_PATH'], path)
        else:
            dest_sc_seg_path = path
        in_sc_seg_path = get_cont_path_from_other_cont(dest_sc_seg_path, cont)
        target_path = get_img_path_from_label_path(dest_sc_seg_path)
        img_path = get_cont_path_from_other_cont(target_path, cont)
        if not os.path.exists(img_path) or not os.path.exists(target_path) or not os.path.exists(in_sc_seg_path) or not os.path.exists(dest_sc_seg_path):
            #raise ValueError(f'Error while loading subject\n {img_path}, {target_path}, {in_sc_seg_path} or {dest_sc_seg_path} might not exist')
            err.append([in_sc_seg_path, 'path error'])
        else:
            # Register and crop contrasts using the SC segmentation
            derivatives_path = os.path.join(dest_sc_seg_path.split('derivatives')[0], 'derivatives/regNcrop')
            errcode, img_path, target_path = registerNcrop(in_path=img_path, dest_path=target_path, in_sc_path=in_sc_seg_path, dest_sc_path=dest_sc_seg_path, derivatives_folder=derivatives_path)
            if errcode[0] != 0:
                err.append([in_sc_seg_path, errcode[1]])
            # Output paths using MONAI load_decathlon_datalist format
            else:
                out_decathlon_monai.append({'image':os.path.abspath(img_path), 'label':os.path.abspath(target_path)})
        
        # Plot progress
        bar.suffix  = f'{paths.index(path)+1}/{len(paths)}'
        bar.next()
    bar.finish()
    return out_decathlon_monai, err