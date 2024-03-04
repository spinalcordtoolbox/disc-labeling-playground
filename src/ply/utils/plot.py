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