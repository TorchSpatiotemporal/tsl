import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tsl.data import SpatioTemporalDataset, SynchMode

# Plotting functions ###############
pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(edgeitems=3, precision=3)
torch.set_printoptions(edgeitems=2, precision=3)


# Utility functions ################
def print_matrix(matrix):
    return pd.DataFrame(matrix)


def print_model_size(model):
    tot = sum([p.numel() for p in model.parameters() if p.requires_grad])
    out = f"Number of model ({model.__class__.__name__}) parameters:{tot:10d}"
    print("=" * len(out))
    print(out)


# Plotting functions ###############

def get_darker_shades(base_color, num_shades, darken_factor=0.2):
    """
    Generate progressively darker shades of a color.

    Parameters:
        base_color (str or tuple): The base color as a color name, hex, or RGB tuple.
        num_shades (int): The number of shades to generate.
        darken_factor (float): The percentage to darken with each step (between 0 and 1).
                               A factor of 0 means no change; 0.2 means each shade is 20% darker.

    Returns:
        list: A list of RGB tuples representing darker shades.
    """
    base_color = mcolors.to_rgba(base_color)
    shades = [base_color]
    darken_factor = 1 - darken_factor
    for i in range(1, num_shades):
        shade = (
            base_color[0] * (darken_factor ** i),  # Red channel
            base_color[1] * (darken_factor ** i),  # Green channel
            base_color[2] * (darken_factor ** i),  # Blue channel
            base_color[3],  # Alpha channel
        )
        shades.append(shade)
    return shades


def plot_inputs_and_target(
        dataset: SpatioTemporalDataset,
        sample_idx: int = 0,
        node_idx: int = 0,
        plot_neighbors: bool = False
) -> None:
    """
    Plots each feature of input tensors on the left, and a single target tensor
    on the right in the first row only.

    Args:
        inputs_dict (dict): Dictionary of input tensors, each of shape
                            [(batch,) time, nodes, features].
        target_tensor (torch.Tensor): Target tensor of shape [(batch,) time, nodes, 1].
        batch_idx (int): Index of the batch to plot if batch dimension is present.
        node_idx (int): Index of the node to plot if node dimension is present.
        mask (Optional[torch.Tensor]): Optional mask tensor with the same shape as
                                       the target tensor, used to set certain target
                                       values to NaN where mask is False.
    """

    data = dataset[sample_idx]

    # Convert all input tensors to numpy arrays
    inputs_list = [
        (name, torch.select(tensor, -2, node_idx).detach().cpu().numpy())
        for name, tensor in data.input.items()
        if data.pattern[name] in {'n f', 't n f'}
    ]
    if plot_neighbors and dataset.edge_index is not None:
        neighbors_mask = dataset.edge_index[1] == node_idx
        neighbors = dataset.edge_index[0][neighbors_mask]
        names = list(map(lambda x: x[0], inputs_list))
        for name in names:
            neigh_data = torch.index_select(data.input[name], -2, neighbors)
            neigh_data = neigh_data.view(-1, neigh_data.size(-2) * neigh_data.size(-1))
            inputs_list.append((f"{name}_neighbors", neigh_data))
    target = data.y
    if data.has_mask:
        target = torch.where(data.mask, target, torch.nan)
    scaled_target = data.transform['y'].transform(target)
    # Convert target tensor to numpy array
    target = target[:, node_idx].detach().cpu().numpy()
    scaled_target = scaled_target[:, node_idx].detach().cpu().numpy()

    num_inputs = max(len(inputs_list), 2)

    # Set up figure with subplots aligned for inputs and one target in the first row
    # Add a gap between the two columns and rows
    fig, axes = plt.subplots(
        num_inputs,
        2,
        figsize=(15, num_inputs * 4),
        gridspec_kw={'wspace': 0.2, 'hspace': 0.3},
    )
    fig.suptitle(
        f"Node {node_idx} - Sample {sample_idx}", fontsize=18, fontweight='bold', y=0.92
    )

    for i, (ax_input, ax_target) in enumerate(axes):
        if i < len(inputs_list):
            input_name, input_data = inputs_list[i]
            is_neighbors = input_name.endswith("_neighbors")
            if is_neighbors:
                input_name = input_name[:-10]
            # Get if data are in window or horizon
            synch_mode = dataset.input_map[input_name].synch_mode

            if synch_mode is SynchMode.STATIC:
                ax_label = "Channels"
                ax_input.bar(range(len(input_data)), input_data, label="Feature")
            else:
                if synch_mode is SynchMode.WINDOW:
                    ax_label = "Window"
                    ticks = range(-input_data.shape[0], 0)  # -dataset.window
                else:  # synch_mode is SynchMode.HORIZON
                    ax_label = "Horizon"
                    ticks = range(0, input_data.shape[0])  # -dataset.horizon

                keys = dataset.input_map[input_name].keys
                feat_groups = {k: getattr(dataset, k).size(-1) for k in keys}
                if is_neighbors:
                    n_features = sum(feat_groups.values())
                    n_neigh = input_data.size(-1) // n_features
                    feat_groups = {k: v * n_neigh for k, v in feat_groups.items()}

                # Plot each feature in the input data, and use same color for each group
                curr_feat = 0
                for curr, group in enumerate(keys):
                    group_size = feat_groups[group]
                    colors = get_darker_shades(f"C{curr}", group_size)
                    ax_input.plot(
                        ticks,
                        input_data[:, curr_feat],
                        label=f"{group} [{group_size}]",
                        color=colors[0],
                    )
                    for j in range(1, group_size):
                        ax_input.plot(
                            ticks, input_data[:, curr_feat + j], color=colors[j]
                        )
                    curr_feat += group_size

            if i == 0:
                ax_input.set_title("Inputs", fontsize=16, fontweight='bold')
            ax_input.set_xlabel(ax_label)
            if is_neighbors:
                input_name = f"{input_name} (neighbors of {node_idx})"
            ax_input.set_ylabel(input_name, fontweight='bold')
            ax_input.spines['top'].set_visible(False)
            ax_input.spines['right'].set_visible(False)
            ax_input.grid(True, alpha=0.5)
            ax_input.legend()
        else:
            ax_input.axis('off')

        # Plot the target data only once in the first row, second column
        if i == 0:
            ax_target.plot(
                range(0, target.shape[0]), target, color='orange', label="Target"
            )
            ax_target.set_title("Target", fontsize=16, fontweight='bold')
            ax_target.set_xlabel("Horizon")
            ax_target.set_xlim(0, target.shape[0])
            ax_target.set_ylabel("y", fontweight='bold')
            ax_target.spines['top'].set_visible(False)
            ax_target.spines['right'].set_visible(False)
            ax_target.grid(True, alpha=0.5)
            ax_target.legend()
        elif i == 1:
            ax_target.plot(
                range(0, scaled_target.shape[0]),
                scaled_target,
                color='orange',
                label="Scaled target",
            )
            ax_target.set_xlabel("Horizon")
            ax_target.set_xlim(0, scaled_target.shape[0])
            ax_target.set_ylabel("y", fontweight='bold')
            ax_target.spines['top'].set_visible(False)
            ax_target.spines['right'].set_visible(False)
            ax_target.grid(True, alpha=0.5)
            ax_target.legend()
        else:  # Turn off the second column for all other inputs
            ax_target.axis('off')

    plt.show()
