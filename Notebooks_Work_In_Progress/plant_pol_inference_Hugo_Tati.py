#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python functions to interact with the Stan model.

Author:
Jean-Gabriel Young <jgyou@umich.edu>
"""
import numpy as np
import pickle
import pystan
import os
import matplotlib.pyplot as plt

abs_path = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Model functions
# =============================================================================
def compile_stan_model(force=False):
    """Autocompile Stan model."""
    source_path = os.path.join(abs_path, 'model.stan')
    target_path = os.path.join(abs_path, 'model.bin')

    if os.path.exists(target_path):
        # Test whether the model has changed and only compile if it did
        with open(target_path, 'rb') as f:
            current_model = pickle.load(f)
        with open(source_path, 'r') as f:
            file_content = "".join([line for line in f])
        if file_content != current_model.model_code or force:
            print(target_path, "[Compiling]", ["", "[Forced]"][force])
            model = pystan.StanModel(source_path, model_name="plant_pol")
            with open(target_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            print(target_path, "[Skipping --- already compiled]")
    else:
        # If model binary does not exist, compile it
        print(target_path, "[Compiling]")
        model = pystan.StanModel(source_path, model_name="plant_pol")
        with open(target_path, 'wb') as f:
            pickle.dump(model, f)


def load_model():
    """Load the model to memory."""
    compile_stan_model()
    with open(os.path.join(abs_path, "model.bin"), 'rb') as f:
        return pickle.load(f)


# =============================================================================
# Sampling functions
# =============================================================================
def generate_sample(M, model, num_chains=4, warmup=5000, num_samples=500):
    """Run sampling for data matrix M."""
    # Prepare the data dictionary
    data = dict()
    data = {"n_p": M.shape[0],
            "n_a": M.shape[1],
            "M": M}
    samples = model.sampling(data=data,
                             chains=4,
                             iter=warmup + num_samples,
                             warmup=warmup,
                             control={'max_treedepth': 15})
    return samples


def save_samples(samples, fpath='samples.bin'):
    """Save samples as binaries, with pickle.

    Warning
    -------
    To retrieve this data, one has to load *the exact version of the model*
    used to generate the samples in memory. Hence, re-compiling the model will
    make the data inaccessible.
    """
    with open(fpath, 'wb') as f:
        pickle.dump(samples, f)


def load_samples(fpath='samples.bin'):
    """Load samples from binaries, with pickle.

    Warning
    -------
    Must have loaded *the same version of the model* in memory.
    """
    with open(fpath, 'rb') as f:
        return pickle.load(f)


def test_samples(samples, tol=0.1, num_chains=4):
    """Verify that no chain has a markedly lower average log-probability."""
    n = len(samples['lp__']) // num_chains  # number of samples per chain
    log_probs = [samples['lp__'][i * n:(i + 1) * n] for i in range(num_chains)]
    log_probs_means = np.array([np.mean(lp) for lp in log_probs])
    return np.all(log_probs_means - (1 - tol) * max(log_probs_means) > 0)


# =============================================================================
# Inference functions
# =============================================================================
def get_posterior_predictive_matrix(samples):
    """Calculate the posterior predictive matrix."""
    Q = samples['Q']
    C = samples['C']
    r = samples['r']
    ones = np.ones((len(samples['lp__']), Q.shape[1], Q.shape[2]))
    sigma_tau = np.einsum('ki,kj->kij', samples['sigma'], samples['tau'])
    accu = (1 - Q) * np.einsum('kij,k->kij', ones, C) * sigma_tau
    accu += Q * np.einsum('kij,k->kij', ones, C * (1 + r)) * sigma_tau
    return np.mean(accu, axis=0)


def estimate_network(samples):
    """Return the matrix of edge probabilities P(B_ij=1)."""
    return np.mean(samples['Q'], axis=0)


def get_network_property_distribution(samples, property, num_net=10):
    """Return the average posterior value of an arbitrary network property.

    Input
    -----
    samples: StanFit object
        The posterior samples.
    property: function
        This function should take an incidence matrix as input and return a
        scalar.
    num_net: int
        Number of networks to generate for each parameter samples.
    """
    values = np.zeros(len(samples['lp__']) * num_net)
    for i, Q in enumerate(samples['Q']):
        for j in range(num_net):
            B = np.random.binomial(n=1, p=Q)
            values[i * num_net + j] = property(B)
    return values


def get_posteriors_predictive_matrices(samples):
    """Calculate the posterior predictive matrix."""
    Q = samples['Q']
    C = samples['C']
    r = samples['r']
    ones = np.ones((len(samples['lp__']), Q.shape[1], Q.shape[2]))
    sigma_tau = np.einsum('ki,kj->kij', samples['sigma'], samples['tau'])
    accu = (1 - Q) * np.einsum('kij,k->kij', ones, C) * sigma_tau
    accu += Q * np.einsum('kij,k->kij', ones, C * (1 + r)) * sigma_tau
    return accu

def figure_1_c(samples,n_a,n_p):
    import matplotlib.gridspec as gridspec
    edge_prob = estimate_network(samples)

    mean_tau = np.mean(samples['tau'], axis=0)
    mean_sigma = np.mean(samples['sigma'], axis=0)

    fig = plt.figure(figsize=(10, 8))
    plt.title("Fig. 1(c)")

    gs = gridspec.GridSpec(2, 3,
                           height_ratios=[1, 4], # Pollinator bars row height is 1, Heatmap row height is 4
                           width_ratios=[4, 1, 0.5], # Heatmap col width is 4, Plant bars col width is 1, Cbar col width is 0.5
                           wspace=0.1, # Horizontal space between heatmap and plant bars
                           hspace=0.1) # Vertical space between pollinator bars and heatmap
    
    
    ax_heatmap = fig.add_subplot(gs[1, 0])
    
    # ax_pollinators will be in the top-left cell, sharing x-axis with heatmap for alignment
    ax_pollinators = fig.add_subplot(gs[0, 0], sharex=ax_heatmap)
    
    # ax_plants will be in the bottom-middle cell, sharing y-axis with heatmap for alignment
    ax_plants = fig.add_subplot(gs[1, 1], sharey=ax_heatmap)
    
    # --- Plot Heatmap ---
    mesh = ax_heatmap.pcolormesh(edge_prob, cmap='Blues', vmin=0, vmax=1)
    
    # Set heatmap ticks and labels
    # Ticks are placed at the center of each cell (0.5, 1.5, ...)
    ax_heatmap.set_xticks(np.arange(n_a) + 0.5)
    
    ax_heatmap.set_yticks(np.arange(n_p) + 0.5)
    # Set heatmap axis labels
    ax_heatmap.set_xlabel('Animals', fontsize=10)
    ax_heatmap.set_ylabel('Animals', fontsize=10)
    
    # Set heatmap limits (important for aligning with shared axes)
    ax_heatmap.set_xlim(0, n_a)
    ax_heatmap.set_ylim(n_p, 0) # Reverse y-axis so plant 1 is at the top
    
    ax_pollinators.bar(np.arange(n_a) + 0.5, mean_tau, color='orange', width=0.8) # width < 1 adds 
    
    ax_pollinators.set_xticks([])
    ax_pollinators.xaxis.set_visible(False) # Ensure x-axis is completely hidden
    
    # Set y-label for pollinator bars
    ax_pollinators.set_ylabel('Effective\nabundance', fontsize=9)
    
    # Set y-limits for pollinator bars (start from 0)
    ax_pollinators.set_ylim(0, mean_tau.max() * 1.1) # Add a little padding above max value
    
    # Remove the bottom spine (the one bordering the heatmap)
    ax_pollinators.spines['bottom'].set_visible(False)
    
    ax_plants.barh(np.arange(n_p) + 0.5, mean_sigma, color='steelblue', height=0.8) # height < 1 adds space
    
    # Hide y-axis ticks and labels as they are shared with heatmap to the left
    ax_plants.set_yticks([])
    ax_plants.yaxis.set_visible(False) # Ensure y-axis is completely hidden
    
    # Set x-label for plant bars
    ax_plants.set_xlabel('Effective\nabundance', fontsize=9)
    
    # Set x-limits for plant bars (start from 0)
    ax_plants.set_xlim(0, mean_sigma.max() * 1.1) # Add a little padding right of max value
    
    # Ensure y-axis limits match heatmap (reversed)
    ax_plants.set_ylim(n_p, 0) # Sharey handles this, but explicit setting doesn't hurt
    
    # Remove the left spine (the one bordering the heatmap)
    ax_plants.spines['left'].set_visible(False)
    
    # --- Adjust appearance ---
    # Hide ticks on the side where axes meet to remove redundancy
    ax_heatmap.tick_params(axis='x', top=False)
    ax_heatmap.tick_params(axis='y', right=False)
    
    ax_pollinators.tick_params(axis='x', bottom=False) # Should already be hidden by set_xticks([])
    
    ax_plants.tick_params(axis='y', left=False) # Should already be hidden by set_yticks([])
    
    
    plt.show()

def plot_subset_interaction_matrices(M, M_tilde, edge_prob, n_p, n_a, subset_n_p, subset_n_a):
    """
    Plots subsets of interaction and edge probability matrices for randomly selected 
    plants and pollinators.

    Args:
        M: Full input interaction matrix.
        M_tilde: Full posterior predictive interaction matrix.
        edge_prob: Full edge probability matrix.
        n_p: Total number of plants.
        n_a: Total number of pollinators.
        subset_n_p: Number of plants to include in the subset plot.
        subset_n_a: Number of pollinators to include in the subset plot.
    """

    if subset_n_p > n_p:
        print(f"Warning: subset_n_p ({subset_n_p}) is larger than n_p ({n_p}). Using n_p instead.")
        subset_n_p = n_p
    if subset_n_a > n_a:
        print(f"Warning: subset_n_a ({subset_n_a}) is larger than n_a ({n_a}). Using n_a instead.")
        subset_n_a = n_a
    if subset_n_p <= 0 or subset_n_a <= 0:
        print("Error: Subset sizes must be positive.")
        return

    # --- Randomly select indices for plants and pollinators ---
    # Use replace=False to ensure unique indices
    plant_indices = np.sort(np.random.choice(range(n_p), size=subset_n_p, replace=False))
    pollinator_indices = np.sort(np.random.choice(range(n_a), size=subset_n_a, replace=False))

    # --- Extract the sub-matrices using the selected indices ---
    # np.ix_ is useful for creating index arrays for sub-matrix selection
    ixgrid = np.ix_(plant_indices, pollinator_indices)
    M_sub = M[ixgrid]
    M_tilde_sub = M_tilde[ixgrid]
    edge_prob_sub = edge_prob[ixgrid]

    # --- Define original labels for the selected subset ---
    plant_labels = plant_indices + 1  # Original identifiers (1-based)
    pollinator_labels = pollinator_indices + 1 # Original identifiers (1-based)

    # --- Helper function for plotting a single matrix subset ---
    def plot_single_subset(matrix_sub, title, colorbar_label, cmap, vmin=None, vmax=None):
        plt.figure(figsize=(6, max(2, 6 * subset_n_p / subset_n_a))) # Adjust aspect ratio
        plt.title(title)
        
        # Use imshow for potentially non-uniform grids or pcolormesh if preferred
        # Using pcolormesh as in the original example
        plt.pcolormesh(matrix_sub, cmap=cmap, vmin=vmin, vmax=vmax)
        
        cb = plt.colorbar(fraction=0.04, pad=0.02, aspect=10)
        cb.ax.get_yaxis().labelpad = 15
        cb.set_label(colorbar_label, rotation=270)
        
        # Set ticks based on the subset size and labels based on original IDs
        plt.yticks(np.arange(subset_n_p) + 0.5, plant_labels)
        plt.xticks(np.arange(subset_n_a) + 0.5, pollinator_labels, rotation=90) # Rotate if many labels
        
        plt.xlabel('Animal identifier')
        plt.ylabel('Animal identifier')
        plt.tight_layout() # Adjust layout to prevent labels overlapping

    # --- Plot the subsets ---
    plot_single_subset(M_sub, 'Input matrix (Subset)', 'Number of interactions', plt.cm.Blues)
    plot_single_subset(M_tilde_sub, 'Posterior predictive (Subset)', 'Number of interactions', plt.cm.Blues)
    plot_single_subset(edge_prob_sub, 'Edge probability (Subset)', 'Edge probability', plt.cm.Blues, vmin=0, vmax=1)

    plt.show()

def estimate_networks(samples):
    """Return the matrix of edge probabilities P(B_ij=1)."""
    return samples['Q']


def compute_nodf(matrix):
    """Computes NODF (Nestedness metric based on Overlap and Decreasing Fill)
    for a binary incidence matrix. Assumes shape (plants × pollinators)."""
    matrix = matrix.astype(int)
    rows, cols = matrix.shape

    # Sort rows and columns by decreasing degree
    row_degrees = matrix.sum(axis=1)
    col_degrees = matrix.sum(axis=0)

    row_order = np.argsort(-row_degrees)
    col_order = np.argsort(-col_degrees)

    M_sorted = matrix[row_order, :][:, col_order]

    # Row NODF
    nodf_rows = 0
    pairs_rows = 0
    for i in range(rows):
        for j in range(i + 1, rows):
            ki, kj = M_sorted[i].sum(), M_sorted[j].sum()
            if ki == kj or kj == 0:
                continue
            overlap = np.logical_and(M_sorted[i], M_sorted[j]).sum()
            nodf_rows += overlap / min(ki, kj)
            pairs_rows += 1

    # Column NODF
    nodf_cols = 0
    pairs_cols = 0
    for i in range(cols):
        for j in range(i + 1, cols):
            ki, kj = M_sorted[:, i].sum(), M_sorted[:, j].sum()
            if ki == kj or kj == 0:
                continue
            overlap = np.logical_and(M_sorted[:, i], M_sorted[:, j]).sum()
            nodf_cols += overlap / min(ki, kj)
            pairs_cols += 1

    # Combine
    total_pairs = pairs_rows + pairs_cols
    if total_pairs == 0:
        return 0.0
    return (nodf_rows + nodf_cols) / total_pairs

# Function to subsample plant rows
def subsample_matrix(M, frac=0.5):
    n_rows = M.shape[0]
    selected = np.random.choice(n_rows, int(n_rows * frac), replace=False)
    return M[selected, :], selected

# Function to compute average edge probabilities
def average_edge_probs(M, C_samples, r_samples, rho_samples, sigma_samples, tau_samples):
    n_samples = len(C_samples)
    n_plants, n_pollinators = M.shape
    Q_sum = np.zeros((n_plants, n_pollinators))

    for k in range(n_samples):
        sigma_tau = np.outer(sigma_samples[k], tau_samples[k])
        mu0 = C_samples[k] * sigma_tau
        log_numerator = np.log(rho_samples[k] + 1e-12) + M * np.log(1 + r_samples[k] + 1e-12) - mu0 * r_samples[k]
        log_numerator = np.clip(log_numerator, -700, 700)
        numerator = np.exp(log_numerator)
        denominator = (1 - rho_samples[k]) + numerator
        Q = numerator / (denominator + 1e-12)
        Q = np.nan_to_num(Q, nan=0.0, posinf=1.0, neginf=0.0)
        Q_sum += np.clip(Q, 0, 1)

    return Q_sum / n_samples


def read_weighted_edgelist_to_sparse_adjacency(filename, delimiter=None, comment_char="#"):
    import scipy.sparse as sparse
    """
    Reads a weighted edgelist from a text file and returns a sparse adjacency matrix.

    Args:
        filename (str): The path to the edgelist file. Assumes each data line
                        has format "node1 node2 weight", separated by whitespace.
        delimiter (str, optional): The delimiter used to separate fields on each line.
                                   Defaults to None (whitespace). Common values: ',', '\t'.
        comment_char (str, optional): Character that indicates a comment line to be skipped.
                                      Defaults to "#". Set to None to disable comment handling.

    Returns:
        scipy.sparse.csr_matrix: A sparse CSR (Compressed Sparse Row) adjacency matrix
                                  representing the directed, weighted graph. Returns None if
                                  the file cannot be read or contains no valid edges.
        int: The maximum node ID found. This is used as the dimension for the square matrix.
    """
    edges = [] # To store (node1, node2, weight)
    max_node_id = 0

    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, start=1): # 1-based line number
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Skip comment lines based on the specified character
                if comment_char and line.startswith(comment_char):
                    # Optional: print skipped comments for verification
                    # print(f"Skipping comment line {line_num}: '{line}'")
                    continue

                # Process data lines
                parts = line.split(delimiter)

                # We expect exactly 3 fields for a weighted edge list (node1, node2, weight)
                if len(parts) != 3:
                    print(f"Warning: Skipping data line {line_num} due to incorrect number of fields (expected 3, got {len(parts)}): '{line}'")
                    continue

                try:
                    # Strip whitespace from each part and convert types
                    node1 = int(parts[0].strip())
                    node2 = int(parts[1].strip())
                    weight = float(parts[2].strip())
                except ValueError:
                    print(f"Warning: Skipping data line {line_num} due to invalid node ID or weight format: '{line}'")
                    continue
                except Exception as e:
                     # Catch any other unexpected errors during parsing
                     print(f"Warning: Skipping data line {line_num} due to unexpected parsing error: '{line}' - {e}")
                     continue

                # Basic validation for node IDs (must be positive for 1-based indexing)
                if node1 <= 0 or node2 <= 0:
                     print(f"Warning: Skipping data line {line_num} due to non-positive node ID: '{line}'")
                     continue

                # Store the edge information
                edges.append((node1, node2, weight))
                # Update max node ID seen to determine matrix dimensions
                max_node_id = max(max_node_id, node1, node2)

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return None, 0 # Return None and 0 nodes on error
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None, 0 # Return None and 0 nodes on error

    if not edges:
        print("No valid data lines found in the file after processing comments and filtering malformed lines.")
        return None, 0 # Return None and 0 nodes if no valid edges were found

    # Determine the size of the adjacency matrix. Use max_node_id
    # to ensure the matrix is large enough to include all node IDs found.
    num_nodes = max_node_id
    print(f"Successfully read {len(edges)} edges.")
    print(f"Max node ID found: {max_node_id}. Creating a {num_nodes}x{num_nodes} matrix.")
    # The comment "% 2137 128 128" suggests 128 is indeed the total number of nodes.

    # Create the sparse adjacency matrix (assuming directed, weighted)
    # Use dtype=float to store weights (the weights are floating point numbers)
    # Use LIL format for efficient step-by-step construction
    adj_matrix = sparse.lil_matrix((num_nodes, num_nodes), dtype=float)

    # Populate the adjacency matrix with weights
    for node1, node2, weight in edges:
        # Adjust node IDs to be 0-based indices for the matrix
        # Based on the 'asym posweighted' comment, assuming a directed edge from node1 to node2
        # with the specified weight.
        adj_matrix[node1 - 1, node2 - 1] = weight

    # Convert the LIL matrix to CSR format for efficient downstream operations
    return adj_matrix.tocsr(), num_nodes # Return the matrix and the number of nodes