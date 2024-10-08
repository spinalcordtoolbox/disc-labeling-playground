import os
import subprocess
import tempfile
import datetime
import numpy as np
import cv2

from ply.data_management.utils import get_img_path_from_label_path, fetch_subject_and_session
from ply.utils.image import Image


## Functions
def img2label(path_label):
    """
    Based on: https://github.com/spinalcordtoolbox/disc-labeling-hourglass
    Convert nifti image to a list of coordinates
    :param path_label: path to the nifti image
    :return: non-zero labels coordinates, image resolution, image shape
    """
    img = Image(path_label).change_orientation('RSP')
    nx, ny, nz, nt, px, py, pz, pt = img.dim
    discs_labels = [list(map(int, list(coord))) for coord in img.getNonZeroCoordinates(sorting='value')]
    discs_labels = [coord for coord in discs_labels if coord[-1] < 26] # Keep only discs labels. Remove labels 49 and 50 that correspond to the pontomedullary groove (49) and junction (50)
    return discs_labels, (px, py, pz), img.data.shape


##
def apply_preprocessing(img_path, dim):
    '''
    Load and apply preprocessing steps on input data
    :param img_path: Path to Niftii image
    '''
    image_in, res_image, shape_image = load_nifti(img_path, dim=dim)
    image = (image_in - np.mean(image_in))/(np.std(image_in)+1e-100)
    image = normalize(image)
    image = image.astype(np.float32)
    return image, res_image, image_in.shape

##
def cropWithSC(in_path, in_sc_path, tmpdir):
    print('Croping input...')
    # Create mask using the SC centerline
    temp_mask_path = os.path.join(tmpdir, 'mask.nii.gz')
    subprocess.check_call(['sct_create_mask',
                        '-i', in_path,
                        '-p', f'centerline,{in_sc_path}',
                        '-size', '70mm',
                        '-f', 'cylinder',
                        '-o', temp_mask_path])
    
    # Crop using the created mask
    temp_in_crop_path = os.path.join(tmpdir, os.path.basename(in_path).split('.nii.gz')[0] + '_desc-crop.nii.gz')
    subprocess.check_call(['sct_crop_image',
                    '-i', in_path,
                    '-m', temp_mask_path,
                    '-o', temp_in_crop_path])
    return temp_in_crop_path

##
def registerNcrop(in_path, dest_path, in_sc_path, dest_sc_path, derivatives_folder, qc=False):
    '''
    Crop and register two images for training
    '''
    # Create output folder
    in_subjectID, in_sessionID, in_filename, in_contrast, in_echoID, in_acquisition = fetch_subject_and_session(in_path)
    dest_subjectID, dest_sessionID, dest_filename, dest_contrast, dest_echoID, dest_acquisition = fetch_subject_and_session(dest_path)
    out_folder = os.path.join(derivatives_folder, in_subjectID, in_sessionID, in_contrast)

    # Create paths for registration
    in_reg_path = os.path.join(out_folder, in_filename.split('.nii.gz')[0] + '_reg' + '.nii.gz')
    in_ones = os.path.join(out_folder, in_filename.split('.nii.gz')[0] + '_ones' + '.nii.gz')
    in_ones_reg = os.path.join(out_folder, in_filename.split('.nii.gz')[0] + '_ones_reg' + '.nii.gz')
    for_warp_path = os.path.join(out_folder, in_filename.split('.nii.gz')[0] + '_forwarp' + '.nii.gz')
    inv_warp_path = os.path.join(out_folder, in_filename.split('.nii.gz')[0] + '_invwarp' + '.nii.gz')

    # Create QC path
    qc_path = os.path.join(derivatives_folder, 'qc')

    # Create paths for cropping
    input_crop_path = os.path.join(out_folder, in_filename.split('.nii.gz')[0] + '_reg_crop' + '.nii.gz')
    dest_crop_path = os.path.join(out_folder, dest_filename.split('.nii.gz')[0] + '_reg_crop' + '.nii.gz')
    mask_path = os.path.join(out_folder, dest_filename.split('.nii.gz')[0] + '_interSCmask' + '.nii.gz')

    # Create output directory
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    if not os.path.exists(input_crop_path) or not os.path.exists(dest_crop_path):
        if not os.path.exists(in_reg_path) or not os.path.exists(in_ones_reg) or not os.path.exists(for_warp_path) or not os.path.exists(inv_warp_path):
            # Set image orientation to RSP
            out=subprocess.run(['sct_image',
                                    '-i', in_path,
                                    '-setorient', 'RSP'])
            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''
            
            out=subprocess.run(['sct_image',
                                    '-i', in_sc_path,
                                    '-setorient', 'RSP'])
            
            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''

            out=subprocess.run(['sct_image',
                                    '-i', dest_sc_path,
                                    '-setorient', 'RSP'])
            
            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''

            out=subprocess.run(['sct_image',
                                    '-i', dest_path,
                                    '-setorient', 'RSP'])

            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''
            
            # Create a coverage mask with ones to know where the spinal cord is present
            out=subprocess.run(['sct_create_mask',
                                '-i', in_path,
                                '-o', in_ones,
                                '-size', '500',
                                '-p', f'centerline,{in_sc_path}'])

            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''

            # Register input_image to destination_image
            out=subprocess.run(['sct_register_multimodal',
                                '-i', in_path,
                                '-d', dest_path,
                                '-iseg', in_sc_path,
                                '-dseg', dest_sc_path,
                                '-param', 'step=1,type=seg,algo=centermass',
                                '-qc', qc_path,
                                '-qc-subject', in_subjectID,
                                '-o', in_reg_path,
                                '-owarp', for_warp_path,
                                '-owarpinv', inv_warp_path])
            
            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''
            
            # Bring coverage to destination space
            out=subprocess.run(['sct_apply_transfo',
                                '-i', in_ones,
                                '-d', dest_path,
                                '-w', for_warp_path,
                                '-x', 'linear',
                                '-o', in_ones_reg])

            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''

        
        if not os.path.exists(mask_path):
            # Create spinalcord mask for cropping see https://spinalcordtoolbox.com/user_section/tutorials/multimodal-registration/contrast-agnostic-registration/preprocessing-t2.html#creating-a-mask-around-the-spinal-cord
            out=subprocess.run(['sct_create_mask',
                                    '-i', in_reg_path,
                                    '-p', f'centerline,{dest_sc_path}',
                                    '-size', '70mm',
                                    '-f', 'cylinder',
                                    '-o', mask_path])

            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''
            
            # Multiply with one reg to extract the shared fov between the 2 images
            out=subprocess.run(['sct_maths',
                                    '-i', mask_path,
                                    '-mul', in_ones_reg,
                                    '-o', mask_path])
            
            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''
        
        # Crop registered contrast
        out=subprocess.run(['sct_crop_image',
                                '-i', in_reg_path,
                                '-m', mask_path,
                                '-o', input_crop_path])
        
        if out.returncode != 0:
            return (1, " ".join(out.args)), '', ''

        # Crop dest contrast
        out=subprocess.run(['sct_crop_image',
                                '-i', dest_path,
                                '-m', mask_path,
                                '-o', dest_crop_path])

        if out.returncode != 0:
            return (1, " ".join(out.args)), '', ''

        # TODO: find a way to crop the warping field to the same dimensions
    else:
        if qc:
            dest_sc_crop_path = os.path.join(dest_folder, dest_filename.split('.nii.gz')[0] + '_sc_crop' + '.nii.gz')
            if not os.path.exists(dest_sc_crop_path):
                subprocess.check_call(['sct_crop_image',
                                '-i', dest_sc_path,
                                '-m', mask_path,
                                '-o', dest_sc_crop_path])
            
            subprocess.check_call(['sct_qc',
                                    '-i', input_crop_path,
                                    '-d', dest_crop_path,
                                    '-s', dest_sc_crop_path,
                                    '-p', 'sct_register_multimodal',
                                    '-qc', qc_path])
    return (0, ''), input_crop_path, dest_crop_path

##
def registerNoSC(in_path, dest_path, derivatives_folder):
    '''
    Crop and register two images for training
    '''
    # Create output folder
    in_subjectID, in_sessionID, in_filename, in_contrast, in_echoID, in_acquisition = fetch_subject_and_session(in_path)
    dest_subjectID, dest_sessionID, dest_filename, dest_contrast, dest_echoID, dest_acquisition = fetch_subject_and_session(dest_path)
    out_folder = os.path.join(derivatives_folder, in_subjectID, in_sessionID, in_contrast)

    # Create paths for registration
    in_reg_path = os.path.join(out_folder, in_filename.split('.nii.gz')[0] + '_reg' + '.nii.gz')
    dest_ones = os.path.join(out_folder, dest_filename.split('.nii.gz')[0] + '_ones' + '.nii.gz')
    in_ones = os.path.join(out_folder, in_filename.split('.nii.gz')[0] + '_ones' + '.nii.gz')
    in_ones_reg = os.path.join(out_folder, in_filename.split('.nii.gz')[0] + '_ones_reg' + '.nii.gz')
    for_warp_path = os.path.join(out_folder, in_filename.split('.nii.gz')[0] + '_forwarp' + '.nii.gz')
    inv_warp_path = os.path.join(out_folder, in_filename.split('.nii.gz')[0] + '_invwarp' + '.nii.gz')

    # Create paths for cropping
    input_crop_path = os.path.join(out_folder, in_filename.split('.nii.gz')[0] + '_reg_crop' + '.nii.gz')
    dest_crop_path = os.path.join(out_folder, dest_filename.split('.nii.gz')[0] + '_reg_crop' + '.nii.gz')
    mask_path = os.path.join(out_folder, dest_filename.split('.nii.gz')[0] + '_interSCmask' + '.nii.gz')

    # Create output directory
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    if not os.path.exists(input_crop_path) or not os.path.exists(dest_crop_path):
        if not os.path.exists(in_reg_path) or not os.path.exists(in_ones_reg) or not os.path.exists(for_warp_path) or not os.path.exists(inv_warp_path):
            # Set image orientation to RSP
            out=subprocess.run(['sct_image',
                                    '-i', in_path,
                                    '-setorient', 'RSP'])
            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''

            out=subprocess.run(['sct_image',
                                    '-i', dest_path,
                                    '-setorient', 'RSP'])

            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''
            
            # Create a coverage mask with ones to know find the shared fov
            out=subprocess.run(['sct_create_mask',
                                '-i', in_path,
                                '-o', in_ones,
                                '-size', '500',
                                '-p', 'center'])

            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''
            
            out=subprocess.run(['sct_create_mask',
                                '-i', dest_path,
                                '-o', dest_ones,
                                '-size', '500',
                                '-p', 'center'])

            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''

            # Register input_image to destination_image
            out=subprocess.run(['sct_register_multimodal',
                                '-i', in_path,
                                '-d', dest_path,
                                '-identity', '1', # No registration is actually performed here just padding
                                '-x', 'nn',
                                '-o', in_reg_path,
                                '-owarp', for_warp_path,
                                '-owarpinv', inv_warp_path])
            
            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''
            
            # Bring coverage to destination space
            out=subprocess.run(['sct_apply_transfo',
                                '-i', in_ones,
                                '-d', dest_path,
                                '-w', for_warp_path,
                                '-x', 'linear',
                                '-o', in_ones_reg])

            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''

        
        if not os.path.exists(mask_path):
            # Multiply with one reg to extract the shared fov between the 2 images
            out=subprocess.run(['sct_maths',
                                    '-i', dest_ones,
                                    '-mul', in_ones_reg,
                                    '-o', mask_path])
            
            if out.returncode != 0:
                return (1, " ".join(out.args)), '', ''
        
        # Set pixels out of this mask to zero
        out=subprocess.run(['sct_maths',
                                '-i', in_reg_path,
                                '-mul', mask_path,
                                '-o', in_reg_path])
        
        if out.returncode != 0:
            return (1, " ".join(out.args)), '', ''
        
        out=subprocess.run(['sct_maths',
                                '-i', dest_path,
                                '-mul', mask_path,
                                '-o', dest_crop_path])
        
        if out.returncode != 0:
            return (1, " ".join(out.args)), '', ''
        
        # Crop registered contrast
        out=subprocess.run(['sct_crop_image',
                                '-i', in_reg_path,
                                '-m', mask_path,
                                '-o', input_crop_path])
        
        if out.returncode != 0:
            return (1, " ".join(out.args)), '', ''

        # Crop dest contrast
        out=subprocess.run(['sct_crop_image',
                                '-i', dest_crop_path,
                                '-m', mask_path,
                                '-o', dest_crop_path])

        if out.returncode != 0:
            return (1, " ".join(out.args)), '', ''

    return (0, ''), input_crop_path, dest_crop_path

##
def load_nifti(path_im, dim='3D', orientation='RSP'):
    """
    Based on: https://github.com/spinalcordtoolbox/disc-labeling-hourglass
    Load a Nifti image using RSP orientation and fetch its metadata.
    :param path_im: path to the nifti image
    :param dim: output format of the image (2D or 3D)
    :return: - image array (in dim format)
             - resolution (in dim format)
             - original shape
             - original orientation
    """
    img = Image(path_im)
    orig_orientation = img.orientation
    img.change_orientation(orientation)
    nx, ny, nz, nt, px, py, pz, pt = img.dim
    if dim == '3D':
        return np.array(img.data), (px, py, pz), img.data.shape, orig_orientation
    elif dim == '2D':
        # Average middle slices to reduce noise
        nb_slice = 1
        while (nb_slice + 2)*px < 3: # To avoid having too blurry images
            nb_slice += 2 # Add 2 to always be symetrical

        arr = np.array(img.data)
        ind = arr.shape[0]//2
        if nb_slice > 1:
            slice_before = (nb_slice-1)//2
            slice_after = nb_slice-1-slice_before
            out_img = np.mean(arr[ind - slice_before : ind + slice_after, :, :], 0)
        else:
            out_img = arr[ind, :, :]
        
        return out_img, (py, pz), img.data.shape, orig_orientation
    else:
        raise ValueError(f'Unknown dimension : {dim}')

##
def normalize(arr):
    '''
    Normalize image
    '''
    ma = arr.max()
    mi = arr.min()
    return ((arr - mi) / (ma - mi + 0.00001))


##
def tuple_type_int(strings):
    '''
    Copied from https://stackoverflow.com/questions/33564246/passing-a-tuple-as-command-line-argument
    '''
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def tuple_type_float(strings):
    '''
    Copied from https://stackoverflow.com/questions/33564246/passing-a-tuple-as-command-line-argument
    '''
    strings = strings.replace("(", "").replace(")", "")
    mapped_float = map(float, strings.split(","))
    return tuple(mapped_float)


##
def tmp_create(basename):
    """Create temporary folder and return its path
    Based on https://github.com/spinalcordtoolbox/spinalcordtoolbox/
    """
    prefix = f"{basename}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    print(f"Creating temporary folder ({tmpdir})")
    return tmpdir


##
def tuple2string(t):
    return str(t).replace(' ', '').replace('(','').replace(')','').replace(',','-')

##
def qc_reg_rgb(image_name, image, target, qc_path):
    '''
    QC registration between image and target
    '''
    image = normalize(image.astype(np.float32))
    target = normalize(target.astype(np.float32))

    nx, ny, nz = image.shape

    out_sag = np.zeros([s*2 for s in (ny, nz)]+[3])
    out_sag[::2,1::2,0]=out_sag[1::2,::2,0]=image[nx//2,:,:]
    out_sag[::2,1::2,1]=out_sag[1::2,::2,1]=image[nx//2,:,:]
    out_sag[::2,::2,2]=out_sag[1::2,1::2,2]=target[nx//2,:,:]
    out_sag[::2,::2,1]=out_sag[1::2,1::2,1]=target[nx//2,:,:]
    qc_sag_path = os.path.join(qc_path, 'sag')
    if not os.path.exists(qc_sag_path):
        os.makedirs(qc_sag_path)
    cv2.imwrite(os.path.join(qc_sag_path, image_name.replace('.nii.gz', '.png')), out_sag*255)
    
    out_ax = np.zeros([s*2 for s in (nx,nz)]+[3])
    out_ax[::2,1::2,0]=out_ax[1::2,::2,0]=image[:,ny//2,:]
    out_ax[::2,1::2,1]=out_ax[1::2,::2,1]=image[:,ny//2,:]
    out_ax[::2,::2,2]=out_ax[1::2,1::2,2]=target[:,ny//2,:]
    out_ax[::2,::2,1]=out_ax[1::2,1::2,1]=target[:,ny//2,:]
    qc_ax_path = os.path.join(qc_path, 'ax')
    if not os.path.exists(qc_ax_path):
        os.makedirs(qc_ax_path)
    cv2.imwrite(os.path.join(qc_ax_path, image_name.replace('.nii.gz', '.png')), out_ax*255)

def qc_side_by_side(image_name, image, target, qc_path):
    '''
    QC registration between image and target
    '''
    image = normalize(image.astype(np.float32))
    target = normalize(target.astype(np.float32))

    nx, ny, nz = image.shape

    out_sag = np.zeros([ny, 2*nz])
    out_sag[:,:nz]= image[nx//2,:,:]
    out_sag[:,nz:]= target[nx//2,:,:]

    qc_sag_path = os.path.join(qc_path, 'sag')
    if not os.path.exists(qc_sag_path):
        os.makedirs(qc_sag_path)
    cv2.imwrite(os.path.join(qc_sag_path, image_name.replace('.nii.gz', '.png')), out_sag*255)
    
    out_ax = np.zeros([nx, 2*nz])
    out_ax[:,:nz]= image[:,ny//2,:]
    out_ax[:,nz:]= target[:,ny//2,:]

    qc_ax_path = os.path.join(qc_path, 'ax')
    if not os.path.exists(qc_ax_path):
        os.makedirs(qc_ax_path)
    cv2.imwrite(os.path.join(qc_ax_path, image_name.replace('.nii.gz', '.png')), out_ax*255)