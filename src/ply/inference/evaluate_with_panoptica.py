import numpy as np
from panoptica import UnmatchedInstancePair, Panoptic_Evaluator, NaiveThresholdMatching
from panoptica.metrics import Metric
import argparse
import os
import json
from progress.bar import Bar
import csv

from ply.utils.image import Image
from ply.utils.plot import save_violin, save_bar

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run cGAN and SPINEPS inference on a JSON config file')
    parser.add_argument('--config', required=True, help='Config JSON file where every image path used for inference must appear in the field TESTING ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--derivative-gt', required=True, help='BIDS derivative folder where the ground truth segmentations are stored (Required)')
    parser.add_argument('--derivative-pred', required=True, help='BIDS derivative folder where the predicted segmentations are stored (Required)')
    parser.add_argument('--suffix-gt', default='_label-spine_dseg', help='BIDS entity + suffix used to name the file (default=_label-vert_dseg)')
    parser.add_argument('--suffix-pred', default='_fakeT2w_label-vert_dseg', help='BIDS entity + suffix used to name the file (default=_label-vert_dseg)')
    parser.add_argument('--res-folder', default='results', type=str, help='Output folder where the results will be stored. (default="results")')
    return parser

def evaluate_seg_panoptica():
    parser = get_parser()
    args = parser.parse_args()

    # Create lists
    no_pred = 0
    fp = []
    fn = []
    global_bin_dsc = [] 
    global_bin_assd = []
    instance_iou = []
    instance_iou_std = []
    instance_dice = []
    instance_dice_std = []
    instance_assd = []
    instance_assd_std = []

    # Load variables
    config_path = os.path.abspath(args.config)
    derivative_gt = args.derivative_gt
    derivative_pred = args.derivative_pred
    suffix_gt = args.suffix_gt
    suffix_pred = args.suffix_pred
    res_folder = os.path.abspath(args.res_folder)

    # Load config data
    # Read json file and create a dictionary
    with open(config_path, "r") as file:
        config_data = json.load(file)
    dict_list = config_data['TESTING']

    # Init progression bar
    bar = Bar(f'Evaluate TESTING split with panoptica', max=len(dict_list))

    for di in dict_list:
        path_image = os.path.join(config_data['DATASETS_PATH'], di['IMAGE'])
        # Create output path
        bids_path = path_image.split('sub-')[0]
        derivative_path_gt = os.path.join(bids_path, derivative_gt)
        derivative_path_pred = os.path.join(bids_path, derivative_pred)

        seg_folder_gt = os.path.join(derivative_path_gt, os.path.dirname(path_image.replace(bids_path,'')))
        seg_folder_pred = os.path.join(derivative_path_pred, os.path.dirname(path_image.replace(bids_path,'')))

        seg_path_gt = os.path.join(seg_folder_gt, os.path.basename(path_image).replace('.nii.gz','')) + suffix_gt + ".nii.gz"
        seg_path_pred = os.path.join(seg_folder_pred, os.path.basename(path_image).replace('.nii.gz','')) + suffix_pred + ".nii.gz"
        if os.path.exists(seg_path_pred):
            ref_masks = Image(seg_path_gt).change_orientation('RSP').data.astype('uint8')
            pred_masks = Image(seg_path_pred).change_orientation('RSP').data.astype('uint8')

            # Remove endplate in prediction because not present in groud truth
            pred_masks[np.where(pred_masks>200)] = 0

            # Compute metrics
            sample = UnmatchedInstancePair(prediction_arr=pred_masks, reference_arr=ref_masks)

            evaluator = Panoptic_Evaluator(
                expected_input=UnmatchedInstancePair,
                instance_matcher=NaiveThresholdMatching(),
            )

            result, debug_data = evaluator.evaluate(sample)
            res_dict = result.to_dict()

            fp.append(res_dict["fp"])
            fn.append(res_dict["fn"])
            global_bin_dsc.append(res_dict["global_bin_dsc"])
            global_bin_assd.append(res_dict["global_bin_assd"])
            instance_iou.append(res_dict["sq"])
            instance_iou_std.append(res_dict["sq_std"])
            instance_dice.append(res_dict["sq_dsc"])
            instance_dice_std.append(res_dict["sq_dsc_std"])
            instance_assd.append(res_dict["sq_assd"])
            instance_assd_std.append(res_dict["sq_assd_std"])
        else:
            no_pred += 1
        # Plot progress
        bar.suffix  = f'{dict_list.index(di)+1}/{len(dict_list)}'
        bar.next()
    bar.finish()
    
    # Plot results

    # Create result directory
    ofolder = os.path.join(res_folder, derivative_pred.replace("derivatives/",""))
    if not os.path.exists(ofolder):
        os.makedirs(ofolder)

    ## Global
    out_global = os.path.join(ofolder, "global.png")
    save_violin([global_bin_dsc, global_bin_assd], outpath=out_global, x_names=['Dice', "Assd"], x_axis='', y_axis='Binary metrics (pixels)')
    
    ## Instance iou
    out_iou = os.path.join(ofolder, "iou.png")
    save_violin([instance_iou, instance_iou_std], outpath=out_iou, x_names=['Mean', "STD"], x_axis='', y_axis='Instances IoU (pixels)')
    
    ## Instance dice
    out_dice = os.path.join(ofolder, "dice.png")
    save_violin([instance_dice, instance_dice_std], outpath=out_dice, x_names=['Mean', "STD"], x_axis='', y_axis='Instances Dice (pixels)')
    
    ## Instance assd
    out_assd = os.path.join(ofolder, "assd.png")
    save_violin([instance_assd, instance_assd_std], outpath=out_assd, x_names=['Mean', "STD"], x_axis='', y_axis='Instances Assd (pixels)')

    ## FP and FN
    out_fp_fn = os.path.join(ofolder, "fp_fn.png")
    save_violin([fp, fn], outpath=out_fp_fn, x_names=['False Positive', 'False Negative'], x_axis='', y_axis='False predictions')

    print(f"{no_pred} missing predictions over {len(dict_list)} files total")
    
    ## Save metrics
    d = {
        'global_dice':global_bin_dsc,
        'global_bin_assd':global_bin_assd,
        'iou':instance_iou,
        'iou_std':instance_iou_std,
        'dice':instance_dice,
        'dice_std':instance_dice_std,
        'assd':instance_assd,
        'assd_std':instance_assd_std,
        'fp':fp,
        'fn':fn,
    }

    csv_path = os.path.join(ofolder, "metrics.csv")
    with open(csv_path, "w") as f:
        key_list = d.keys()
        w = csv.writer(f)
        w.writerow(key_list)
        list_len = len(list(d.values())[0])
        for i in range(list_len):
            w.writerow([d[k][i] for k in key_list])


if __name__=='__main__':
    evaluate_seg_panoptica()