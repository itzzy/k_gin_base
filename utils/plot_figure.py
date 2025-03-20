import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
from medutils.visualization import center_crop
from scipy.interpolate import interp1d
from matplotlib import gridspec
import matplotlib.ticker as mticker

## 可视化对比方法--黄文麒师兄
# Load configuration
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load reconstruction data from directory
def load_reconstruction(dir_path, target_t=None):
    file_path = os.path.join(dir_path, 'recon.npy')
    def normalize_image(image):
        abs_image = np.abs(image)
        return (abs_image - abs_image.min()) / (abs_image.max() - abs_image.min())
    file_path_2 = os.path.join(dir_path, 'recon_150.npy')
    if os.path.exists(file_path):
        data = np.load(file_path)
        if target_t is not None and data.shape[0] != target_t:
            time_points = np.linspace(0, data.shape[0] - 1, target_t)
            interp_func = interp1d(np.arange(data.shape[0]), data, axis=0, kind='nearest', fill_value='extrapolate')
            data = interp_func(time_points)

        return normalize_image(data)
    elif os.path.exists(file_path_2):
        data = np.load(file_path_2)
        if target_t is not None and data.shape[0] != target_t:
            time_points = np.linspace(0, data.shape[0] - 1, target_t)
            interp_func = interp1d(np.arange(data.shape[0]), data, axis=0, kind='nearest', fill_value='extrapolate')
            data = interp_func(time_points)
        return normalize_image(data)
    else:
        raise FileNotFoundError(f"File {file_path} not found")

# Main function for visualization
def visualize_results(config_path):
    # Load config
    # config = load_config(config_path)['default']
    config = load_config(config_path)['good1']
    
    # Load reference reconstruction if available
    ref_data = None
    target_t = max([load_reconstruction(os.path.join(dir, config['case_id'])).shape[0] for dir in config['comparison_directories']]) if config['reference_directory'] == 'none' else load_reconstruction(os.path.join(config['reference_directory'], config['case_id'])).shape[0]
    if config['reference_directory'] != 'none':
        ref_data = load_reconstruction(os.path.join(config['reference_directory'], config['case_id']), target_t)
    
    # Load comparison data
    comp_data_list = [load_reconstruction(os.path.join(dir, config['case_id']), target_t) for dir in config['comparison_directories']]
    comp_titles = config.get('comparison_titles', [f'Comparison {i + 1}' for i in range(len(comp_data_list))])

    # Crop ref_data and comp_data_list to the same size of the last one
    img_size = comp_data_list[-1].shape[1:]
    ref_data = center_crop(ref_data, img_size) if ref_data is not None else None
    comp_data_list = [center_crop(data, img_size) for data in comp_data_list]
    
    # Extract configuration settings
    phase_index = config['phase_index']
    y_profile_index = config['y_profile_index']
    roi_coordinates = config['roi_coordinates']
    poi_coordinates = config['poi_coordinates']
    t_poi_coordinates = config['t_poi_coordinates']
    t_arrow_direction = config.get('t_arrow_direction', [5, 0])  # Default direction is pointing right
    arrow_direction = config.get('arrow_direction', [0, -5])  # Default direction is pointing up
    display_range = config['display_range']
    error_display_range = config['error_display_range']
    
    # Calculate figure size dynamically based on image size and number of columns
    num_columns = len(comp_data_list) + (1 if ref_data is not None else 0)
    xy_aspect_ratio = img_size[1] / img_size[0]  # Width / Height for x-y image
    nt = comp_data_list[0].shape[0]  # Time dimension (number of frames)
    
    # Calculate row heights for each type of subplot
    if ref_data is not None:
        row_heights = [img_size[0], img_size[0], img_size[0], nt, nt]
    else:
        row_heights = [img_size[0], img_size[0], nt]
    total_height = sum(row_heights) / 50 
    total_width = (img_size[1] * num_columns) / 50
    
    # Prepare figure
    num_rows = len(row_heights)
    fig = plt.figure(figsize=(total_width, total_height))
    spec = gridspec.GridSpec(num_rows, num_columns + 1, width_ratios=[1] * num_columns + [0.05], height_ratios=row_heights, figure=fig)
    axs = [[fig.add_subplot(spec[row, col]) for col in range(num_columns)] for row in range(num_rows)]
    error_color_ax = None
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    
    # Helper function for adding subplots
    def plot_image(ax, image, vmin, vmax, text=None, cmap='gray', roi=None, profile_line=None):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis('off')
        if text:
            ax.text(0.02, 0.98, text, fontsize=14, color='white', ha='left', va='top', transform=ax.transAxes)
        if roi is not None:
            rect = plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0], roi[3] - roi[1], linewidth=1, edgecolor='darkgoldenrod', linestyle='--', facecolor='none', alpha=0.7)
            ax.add_patch(rect)
        if profile_line is not None:
            ax.axhline(y=profile_line, color='lightsalmon', linestyle='--', linewidth=1, alpha=0.7)
        return im
    
    # Plot reference if available
    images = []
    if ref_data is not None:
        images.append(plot_image(axs[0][0], ref_data[phase_index], vmin=display_range[0], vmax=display_range[1], text='Reference', roi=roi_coordinates, profile_line=y_profile_index))
        plot_image(axs[1][0], ref_data[phase_index, roi_coordinates[1]:roi_coordinates[3], roi_coordinates[0]:roi_coordinates[2]], vmin=display_range[0], vmax=display_range[1])
        axs[1][0].arrow(poi_coordinates[0] - roi_coordinates[0] - arrow_direction[0], poi_coordinates[1] - roi_coordinates[1] - arrow_direction[1], arrow_direction[0], arrow_direction[1], head_width=3, head_length=5, fc='yellow', ec='yellow')
        # disable error map for reference
        plot_image(axs[2][0], np.ones_like(ref_data[phase_index]), vmin=error_display_range[0], vmax=error_display_range[1])
        plot_image(axs[3][0], ref_data[:, y_profile_index, :], vmin=display_range[0], vmax=display_range[1])
        axs[3][0].arrow(t_poi_coordinates[0] - t_arrow_direction[0], t_poi_coordinates[1] - t_arrow_direction[1], t_arrow_direction[0], t_arrow_direction[1], head_width=3, head_length=5, fc='red', ec='red')
        plot_image(axs[4][0], np.ones_like(ref_data[:, y_profile_index, :]), vmin=error_display_range[0], vmax=error_display_range[1])
    
    # Plot comparison results
    for idx, comp_data in enumerate(comp_data_list):
        col_idx = idx + (1 if ref_data is not None else 0)
        images.append(plot_image(axs[0][col_idx], comp_data[phase_index], vmin=display_range[0], vmax=display_range[1], text=comp_titles[idx], roi=roi_coordinates, profile_line=y_profile_index))
        
        # Plot ROI area
        plot_image(axs[1][col_idx], comp_data[phase_index, roi_coordinates[1]:roi_coordinates[3], roi_coordinates[0]:roi_coordinates[2]], vmin=display_range[0], vmax=display_range[1])
        axs[1][col_idx].arrow(poi_coordinates[0] - roi_coordinates[0] - arrow_direction[0], poi_coordinates[1] - roi_coordinates[1] - arrow_direction[1], arrow_direction[0], arrow_direction[1], head_width=3, head_length=5, fc='yellow', ec='yellow')
        
        # Plot error map if reference is available
        if ref_data is not None:
            error_map = np.abs(ref_data[phase_index, roi_coordinates[1]:roi_coordinates[3], roi_coordinates[0]:roi_coordinates[2]] - comp_data[phase_index, roi_coordinates[1]:roi_coordinates[3], roi_coordinates[0]:roi_coordinates[2]])
            images.append(plot_image(axs[2][col_idx], error_map, vmin=error_display_range[0], vmax=error_display_range[1], cmap='viridis'))
        
        # Plot X-T profile
        plot_image(axs[3 if ref_data is not None else 2][col_idx], comp_data[:, y_profile_index, :], vmin=display_range[0], vmax=display_range[1])
        axs[3 if ref_data is not None else 2][col_idx].arrow(t_poi_coordinates[0] - t_arrow_direction[0], t_poi_coordinates[1] - t_arrow_direction[1], t_arrow_direction[0], t_arrow_direction[1], head_width=3, head_length=5, fc='red', ec='red')
        
        # Plot error map of X-T profile if reference is available
        if ref_data is not None:
            error_xt = np.abs(ref_data[:, y_profile_index, :] - comp_data[:, y_profile_index, :])
            plot_image(axs[4][col_idx], error_xt, vmin=error_display_range[0], vmax=error_display_range[1], cmap='viridis')
    
    # Add color bars
    if ref_data is not None:
        # Calculate color bar positions based on GridSpec
        color_ax = fig.add_subplot(spec[:2, -1])
        cbar_img = fig.colorbar(images[0], cax=color_ax, orientation='vertical', ticks=np.linspace(display_range[0], display_range[1], num=5), extend='max')
        cbar_img.ax.tick_params(pad=5, which='both', direction='in')
        cbar_img.outline.set_visible(False)
        # cbar_img.set_label('Image Intensity', fontsize=12, labelpad=5)
        
        error_color_ax = fig.add_subplot(spec[2:, -1])
        cbar_err = fig.colorbar(images[-1], cax=error_color_ax, orientation='vertical', ticks=np.linspace(error_display_range[0], error_display_range[1], num=5), extend='max')
        cbar_err.ax.tick_params(pad=5, which='both', direction='in')
        cbar_err.outline.set_visible(False)
        # cbar_err.set_label('Error Intensity', fontsize=12, labelpad=5)
    else:
        color_ax = fig.add_subplot(spec[:, -1])
        cbar_img = fig.colorbar(images[0], cax=color_ax, orientation='vertical', ticks=np.linspace(display_range[0], display_range[1], num=5), extend='max')
        cbar_img.ax.tick_params(pad=5, which='both', direction='in')
        cbar_img.outline.set_visible(False)
        # cbar_img.set_label('Image Intensity', fontsize=12, labelpad=1)
    
    # plt.show()
    # plt.savefig('comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('comparison.png', dpi=300, bbox_inches='tight')

# Run the visualization
if __name__ == "__main__":
    config_path = 'figconfig.yml'  # Path to your config file
    visualize_results(config_path)
