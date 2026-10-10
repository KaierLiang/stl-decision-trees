import math
import time
import numpy as np


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0  # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference


TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator


def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)
    return tempTimeInterval


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def generate_nodes(depth):
    # Total number of nodes in a complete binary tree up to a given depth
    total_nodes = 2 ** (depth + 1) - 1

    # Number of leaf nodes at the deepest level
    leaf_nodes_count = 2 ** depth

    # All nodes before the last level are decision nodes
    decision_nodes = list(range(total_nodes - leaf_nodes_count))

    # All nodes at the last level are leaf nodes
    leaf_nodes = list(range(total_nodes - leaf_nodes_count, total_nodes))

    return total_nodes, decision_nodes, leaf_nodes


def get_rho_git(time_bound, x, operator):
    lb, ub = time_bound
    temp_x = x[lb:ub + 1]
    if operator == 1:
        return max(temp_x)
    else:
        return -1 * min(temp_x)


def get_discrete_git(time_bound, x, operator):
    lb, ub = time_bound
    temp_x = x[lb:ub + 1]
    if operator == 1:
        return max(temp_x)
    else:
        return min(temp_x)

def get_discrete_feature(x):
    return max(x) # 0 or 1

def generate_combinations(n, m, window_interval=61):
    # Generate combinations
    combinations = [(lb, ub) for lb in range(n, m + 1) for ub in range(lb, m + 1) if ub - lb <= window_interval]
    return combinations


def get_child_nodes(k, n):
    left_child = 2 * k
    right_child = 2 * k + 1
    children = []
    if left_child <= n:
        children.append(left_child)
    if right_child <= n:
        children.append(right_child)
    return children if children else None


def get_formula_for_leaf(leaf_number, formula_true, formula_false):
    # Initialize the formula as an empty string
    formula = ""
    node = leaf_number

    while node > 0:
        parent = (node - 1) // 2

        # Determine if the current node is left or right child and append the formula accordingly
        if node % 2 == 0:
            # It is a right child since it's an even number after subtracting 1
            formula_part = formula_false[parent]
        else:
            # It is a left child
            formula_part = formula_true[parent]

        # Prepend the new part to the formula string
        formula = f"({formula_part}) ^ ({formula})" if formula else formula_part

        # Move up to the parent node for the next iteration
        node = parent

    return formula


def interpolate_data(data, factor):
    # Interpolate the data to increase its size by the specified factor
    new_length = data.shape[2] * factor - (factor - 1)
    interpolated_data = np.zeros((data.shape[0], data.shape[1], new_length))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x = np.linspace(0, data.shape[2] - 1, data.shape[2])  # Original points
            xp = np.linspace(0, data.shape[2] - 1, new_length)  # Interpolated points
            fp = data[i, j, :]
            interpolated_data[i, j, :] = np.interp(xp, x, fp)

    return interpolated_data


def data_sample(data, frequency=1):
    if frequency > 0:
        # Sample the data at the given positive frequency
        return data[:, :, ::frequency]
    elif frequency < 0:
        # Interpolate the data to increase its size
        return interpolate_data(data, -frequency)
    else:
        # Frequency 0 is not valid, return original data
        return data

# def augment_data_with_label_feature(data, labels):
#     """
#     Keep only normal data, copy it to create abnormal data, and add the label as a new feature.
#     """
#     # Keep only normal data
#     normal_indices = np.where(labels == 1)[0]
#     normal_data = data[normal_indices]
#     n_samples, n_dims, n_timesteps = normal_data.shape
#
#     # Create copies of normal data
#     copied_data = normal_data.copy()
#
#     # Create labels for original and copied data
#     original_labels = np.ones(n_samples)  # Normal
#     copied_labels = np.zeros(n_samples)  # Abnormal
#
#     # Add label as a new feature
#     original_feature = np.ones((n_samples, 1, n_timesteps))
#     copied_feature = np.zeros((n_samples, 1, n_timesteps))
#
#     # Augment the data with the new feature
#     augmented_original = np.concatenate((normal_data, original_feature), axis=1)
#     augmented_copied = np.concatenate((copied_data, copied_feature), axis=1)
#
#     # Combine original and copied data
#     final_augmented_data = np.vstack((augmented_original, augmented_copied))
#     final_labels = np.hstack((original_labels, copied_labels))
#
#     # Create a mapping of new indices to original indices
#     index_mapping = list(normal_indices) + list(normal_indices)
#
#     return final_augmented_data, final_labels


def augment_data_with_new_feature(data, labels):
    """
    Augment data with a new feature and create copies of normal data with the new feature set to 1.
    Maintain original indices for the copied data.
    """
    n_samples, n_dims, n_timesteps = data.shape

    # Initialize the new feature
    new_feature = np.zeros((n_samples, 1, n_timesteps))

    # Augment the original data with the new feature (all zeros)
    augmented_data = np.concatenate((data, new_feature), axis=1)
    augmented_labels = labels.copy()

    # Find normal trajectories
    normal_indices = np.where(labels == 1)[0]

    # only use half of the normal data
    normal_indices = normal_indices[:len(normal_indices)//2]

    # Create copies of normal data
    copied_data = data[normal_indices].copy()
    copied_feature = np.ones((len(normal_indices), 1, n_timesteps))
    copied_augmented_data = np.concatenate((copied_data, copied_feature), axis=1)

    # Create labels for copied data (set to abnormal)
    copied_labels = np.zeros(len(normal_indices))

    # Combine original and copied data
    final_augmented_data = np.vstack((augmented_data, copied_augmented_data))
    final_labels = np.hstack((augmented_labels, copied_labels))

    # Create a mapping of new indices to original indices
    index_mapping = list(range(n_samples)) + list(normal_indices)

    return final_augmented_data, final_labels, index_mapping


def build_complete_tree(data_results):
    max_node = max(data_results.keys())
    tree_structure = {}
    for node in range(max_node + 1):
        if node in data_results:
            tree_structure[node] = data_results[node]
        else:
            tree_structure[node] = 0
    return tree_structure


def print_tree_structure(tree_structure, node=0, depth=0):
    if node >= len(tree_structure):
        return

    samples = tree_structure[node]

    # Determine if it's a leaf node (label) based on its children
    left_child = 2 * node + 1
    right_child = 2 * node + 2
    is_leaf = left_child >= len(tree_structure) or right_child >= len(tree_structure)

    indent = "  " * depth
    if is_leaf:
        print(f"{indent}Node {node} : label {samples} (samples: {samples})")
    else:
        print(f"{indent}Node {node} : (samples: {samples})")

        # Print child labels if they exist
        if left_child < len(tree_structure):
            print(f"{indent}  Left child: Node {left_child} (label: {tree_structure[left_child]})")
        if right_child < len(tree_structure):
            print(f"{indent}  Right child: Node {right_child} (label: {tree_structure[right_child]})")

    # Recursively print children
    print_tree_structure(tree_structure, left_child, depth + 1)
    print_tree_structure(tree_structure, right_child, depth + 1)