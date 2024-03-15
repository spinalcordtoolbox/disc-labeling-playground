import numpy as np
import matplotlib.pyplot as plt

def save_bar(names, values, output_path, x_axis, y_axis):
    '''
    Create a histogram plot
    :param names: String list of the names
    :param values: Values associated with the names
    :param output_path: Output path (string)
    :param x_axis: x-axis name
    :param y_axis: y-axis name

    '''
            
    # Set position of bar on X axis
    fig = plt.figure(figsize = (len(names)//2, 5))
 
    # creating the bar plot
    plt.bar(names, values, width = 0.4)
    
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.xticks(names)
    plt.title("Discs distribution")
    plt.savefig(output_path)


def plot_discs_distribution(discs_labels_list, out_path):
    plot_discs = {}
    for discs_list in discs_labels_list:
        for disc_coords in discs_list:
            num_disc = disc_coords[-1]
            if not num_disc in plot_discs.keys():
                plot_discs[num_disc] = 1
            else:
                plot_discs[num_disc] += 1
    # Sort dict
    plot_discs = dict(sorted(plot_discs.items()))
    names, values = list(plot_discs.keys()), list(plot_discs.values())
    # Plot distribution
    save_bar(names, values, out_path, x_axis='Discs number', y_axis='Quantity')


def get_validation_image(in_img, target_img, pred_img):
    in_img = in_img.data.cpu().numpy()
    target_img = target_img.data.cpu().numpy()
    pred_img = pred_img.data.cpu().numpy()
    in_all = []
    target_all = []
    pred_all = []
    for num_batch in range(in_img.shape[0]):
        # Load 3D numpy array
        x = in_img[num_batch, 0]
        y = target_img[num_batch, 0]
        y_pred = pred_img[num_batch, 0]
        shape = x.shape

        # Extract middle slice
        x = x[shape[0]//2,:,:]
        y = y[shape[0]//2,:,:]
        y_pred = y_pred[shape[0]//2,:,:]

        # Normalize intensity
        x = x/np.max(x)*255
        y = y/np.max(y)*255
        y_pred = y_pred/np.max(y_pred)*255

        # Regroup batch
        in_all.append(x)
        target_all.append(y)
        pred_all.append(y_pred)
    
    # Regroup batch into 1 array
    in_line_arr = np.concatenate(np.array(in_all), axis=1)
    target_line_arr = np.concatenate(np.array(target_all), axis=1)
    pred_line_arr = np.concatenate(np.array(pred_all), axis=1)

    # Regroup image/target/pred into 1 array
    img_result = np.concatenate((in_line_arr, target_line_arr, pred_line_arr), axis=0)
    
    return img_result, target_line_arr, pred_line_arr