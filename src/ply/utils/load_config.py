import os
from progress.bar import Bar
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import shutil

from ply.data_management.utils import get_img_path_from_label_path, get_cont_path_from_other_cont, fetch_subject_and_session
from ply.utils.utils import img2label, apply_preprocessing, registerNcrop, registerNoSC, normalize
from ply.utils.plot import plot_discs_distribution
from ply.utils.image import Image
from ply.utils.plot import save_violin

## Functions
def fetch_array_from_config_classifier(config_data, fov=None, dim='3D', split='TRAINING'):
    '''
    This function output 5 lists corresponding to:
        
    
    :param config_data: Config dict where every label used for TRAINING, VALIDATION and/or TESTING has its path specified
    :param fov: Size of the fov that will be used for training. Images bigger than this fov will be added multiple times for the training.
    :param dim: Input dimensions that will be used for training.
    :param split: Split of the data needed in the config file ('TRAINING', 'VALIDATION', 'TESTING').
    :return: - Nifti (dim) numpy array in the RSP orientation
             - the discs labels
             - the subjects names
             - image resolutions
             - the image shape
    '''

    # Check config type to ensure that labels paths are specified and not images
    if config_data['TYPE'] != 'LABEL':
        raise ValueError('TYPE LABEL not detected: PLZ specify paths to labels for training in config file')
    
    # Get file paths based on split
    paths = config_data[split]
    
    # Init progression bar
    bar = Bar(f'Load {split} data with pre-processing', max=len(paths))
    
    imgs = []
    discs_labels_list = []
    subjects = []
    shapes = []
    resolutions = []
    problematic_gt = []
    for path in paths:
        if 'DATASETS_PATH' in config_data.keys():
            label_path = os.path.join(config_data['DATASETS_PATH'], path)
        else:
            label_path = path
        img_path = get_img_path_from_label_path(label_path)
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f'Error while loading subject\n {img_path} or {label_path} might not exist')
        else:
            # Applying preprocessing steps
            image, res_image, shape_image = apply_preprocessing(img_path, dim=dim)
            discs_labels, res_target, shape_target = img2label(target_path)

            # Check for mismatchs between image and target
            if res_image != res_target or shape_image != shape_target:
                raise ValueError(f'Image {img_path} and target {target_path} have different shapes or resolutions')
            
            # Calculate number of images to add based on cropped fov
            if not fov is None:
                Y, X = round(fov[1]/res_image[0]), round(fov[0]/res_image[1])
                nb_same_img = shape_image[0]//Y + shape_image[0]//X + 1
            else:
                nb_same_img = 1
            
            # Check if the disc list is not empty or discs are missing
            if discs_labels and (max(np.array(discs_labels)[:,-1])+1-min(np.array(discs_labels)[:,-1]) == len(np.array(discs_labels))) and (np.array(discs_labels)[:,1] == np.sort(np.array(discs_labels)[:,1])).all():
                # Add the same data multiple when random fov/crop is used
                for i in range(nb_same_img):
                    imgs.append(image)
                    discs_labels_list.append(discs_labels)
                    subject, sessionID, filename, contrast, echoID, acquisition = fetch_subject_and_session(img_path)
                    subjects.append(subject)
                    resolutions.append(res_image)
                    shapes.append(shape_image)
            else:
                problematic_gt.append(label_path)
        
        # Plot progress
        bar.suffix  = f'{paths.index(path)+1}/{len(paths)}'
        bar.next()
    bar.finish()

    # plot discs distribution
    plot_discs_distribution(discs_labels_list, out_path=f'discs_distribution_{split}.png')

    if problematic_gt:
        print("Error with these ground truth\n" + '\n'.join(problematic_gt))
    return imgs, masks, discs_labels_list, subjects, resolutions, shapes


##
def fetch_and_register_config_cGAN(config_data, split='TRAINING', qc=True):
    '''
    :param config_data: Config dict where every label used for TRAINING, VALIDATION and/or TESTING has its path specified
    :param split: Split of the data needed in the config file ('TRAINING', 'VALIDATION', 'TESTING').
    :return: out_decathlon_monai: list of dictionary with image and label paths (like monai load_decathlon_datalist)
        [
            {'image': '/workspace/data/chest_19.nii.gz',  'label': '/workspace/data/chest_19_label.nii.gz'},
            {'image': '/workspace/data/chest_31.nii.gz',  'label': '/workspace/data/chest_31_label.nii.gz'}
        ]
    '''
    # Plot lists
    if qc:
        R, S, P, pR, pS, pP = [], [], [], [], [], []

    # Check config type to ensure that labels paths are specified and not images
    if config_data['TYPE'] != 'CONTRAST':
        raise ValueError('TYPE error: Type CONTRAST not detected')
    
    # Get file paths based on split
    dict_list = config_data[split]
    
    # Init progression bar
    bar = Bar(f'Load {split} data with pre-processing', max=len(dict_list))
    
    err = []
    out_decathlon_monai = []
    for di in dict_list:
        input_img_path = os.path.join(config_data['DATASETS_PATH'], di['INPUT_IMAGE'])
        target_img_path = os.path.join(config_data['DATASETS_PATH'], di['TARGET_IMAGE'])
        derivatives_path = os.path.join(config_data['DATASETS_PATH'], di['INPUT_IMAGE'].split('/')[0], 'derivatives/regNcrop_NoSC')
        if not os.path.exists(input_img_path) or not os.path.exists(target_img_path):
            err.append([input_img_path, 'path error'])
        else:
            # Register
            errcode, target_path, img_path = registerNoSC(in_path=target_img_path, dest_path=input_img_path, derivatives_folder=derivatives_path)
        if errcode[0] != 0:
            err.append([input_img_path, errcode[1]])
        # Output paths using MONAI load_decathlon_datalist format
        else:
            if qc:
                img = Image(img_path).change_orientation('RSP')
                target = Image(target_path).change_orientation('RSP')

                img.data = normalize(img.data.astype(np.float32))
                target.data = normalize(target.data.astype(np.float32))

                nx, ny, nz, nt, px, py, pz, pt = img.dim

                test = np.zeros([s*2 for s in img.data.shape[1:]]+[3])
                test[::2,1::2,0]=test[1::2,::2,0]=img.data[nx//2,:,:]
                test[::2,1::2,1]=test[1::2,::2,1]=img.data[nx//2,:,:]
                test[::2,::2,2]=test[1::2,1::2,2]=target.data[nx//2,:,:]
                test[::2,::2,1]=test[1::2,1::2,1]=target.data[nx//2,:,:]
                qc_sag_path = os.path.join(derivatives_path,'qc', 'sag')
                if not os.path.exists(qc_sag_path):
                    os.makedirs(qc_sag_path)
                cv2.imwrite(os.path.join(qc_sag_path, os.path.basename(img_path.replace('.nii.gz', '.png'))), test*255)
                
                test = np.zeros([s*2 for s in (img.data.shape[0],img.data.shape[-1])]+[3])
                test[::2,1::2,0]=test[1::2,::2,0]=img.data[:,ny//2,:]
                test[::2,1::2,1]=test[1::2,::2,1]=img.data[:,ny//2,:]
                test[::2,::2,2]=test[1::2,1::2,2]=target.data[:,ny//2,:]
                test[::2,::2,1]=test[1::2,1::2,1]=target.data[:,ny//2,:]
                qc_ax_path = os.path.join(derivatives_path,'qc', 'ax')
                if not os.path.exists(qc_ax_path):
                    os.makedirs(qc_ax_path)
                cv2.imwrite(os.path.join(qc_ax_path, os.path.basename(img_path.replace('.nii.gz', '.png'))), test*255)

                R.append(nx)
                S.append(ny)
                P.append(nz)
                pR.append(px)
                pS.append(py)
                pP.append(pz)
            out_decathlon_monai.append({'image':os.path.abspath(img_path), 'label':os.path.abspath(target_path)})
            # Add output if training set
            if split == 'TRAINING':
                out_decathlon_monai.append({'image':os.path.abspath(target_path), 'label':os.path.abspath(target_path)})
        
        # Plot progress
        bar.suffix  = f'{dict_list.index(di)+1}/{len(dict_list)}'
        bar.next()
    bar.finish()
    if qc:
        # Save plot
        save_violin([R,S,P], 'size.png', x_names=['R','S','P'], x_axis='axis', y_axis='size (pixel)')
        save_violin([pR,pS,pP], 'res.png', x_names=['R','S','P'], x_axis='axis', y_axis='resolution (mm/pixel)')
    return out_decathlon_monai, err


def fetch_image_config_cGAN(config_data, split='TRAINING', qc=False):
    '''
    :param config_data: Config dict where every label used for TRAINING, VALIDATION and/or TESTING has its path specified
    :param split: Split of the data needed in the config file ('TRAINING', 'VALIDATION', 'TESTING').
    :return: out_decathlon_monai: list of dictionary with image and label paths (like monai load_decathlon_datalist)
        [
            {'image': '/workspace/data/chest_19.nii.gz',  'label': '/workspace/data/chest_19_label.nii.gz'},
            {'image': '/workspace/data/chest_31.nii.gz',  'label': '/workspace/data/chest_31_label.nii.gz'}
        ]
    '''
    # Plot lists
    if qc:
        R, S, P, pR, pS, pP = [], [], [], [], [], []

    # Check config type to ensure that labels paths are specified and not images
    if config_data['TYPE'] != 'IMAGE':
        raise ValueError('TYPE error: Type IMAGE not detected')
    
    # Get file paths based on split
    dict_list = config_data[split]
    
    # Init progression bar
    bar = Bar(f'Load {split} data with pre-processing', max=len(dict_list))
    
    err = []
    out_decathlon_monai = []
    for di in dict_list:
        input_img_path = os.path.join(config_data['DATASETS_PATH'], di['IMAGE'])
        if not os.path.exists(input_img_path):
            err.append([input_img_path, 'path error'])
        # Output paths using MONAI load_decathlon_datalist forma
        if qc:
            img = Image(input_img_path).change_orientation('RSP')
            nx, ny, nz, nt, px, py, pz, pt = img.dim
            R.append(nx)
            S.append(ny)
            P.append(nz)
            pR.append(px)
            pS.append(py)
            pP.append(pz)
        if split != 'TESTING':
            out_decathlon_monai.append({'image':os.path.abspath(input_img_path), 'label':os.path.abspath(input_img_path)})
        else:
            out_decathlon_monai.append({'image':os.path.abspath(input_img_path)})
        
        # Plot progress
        bar.suffix  = f'{dict_list.index(di)+1}/{len(dict_list)}'
        bar.next()
    bar.finish()
    if qc:
        # Save plot
        save_violin([R,S,P], 'size.png', x_names=['R-L','S-I','P-A'], x_axis='axis', y_axis='size (pixel)')
        save_violin([pR,pS,pP], 'res.png', x_names=['R-L','S-I','P-A'], x_axis='axis', y_axis='resolution (mm/pixel)')
    return out_decathlon_monai, err
