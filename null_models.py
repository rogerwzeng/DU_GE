"""
Null Models for Dissipative Urbanism Validation

This module implements two key null models for validating thermodynamic findings:

1. Null Model 1: Equilibrium (Randomized Flows)
   - Randomize migration flows preserving marginal totals
   - Test if σ_obs > σ_null (entropy production greater than equilibrium)
   
2. Null Model 2: Substantialist (Static Gravity Model)
   - Static gravity model: Flows proportional to pop_i × pop_j / distance_ij^beta
   - Test if geodesic distances explain variance beyond geographic distance
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional, Callable
import warnings


def iterative_proportional_fitting(
    seed_matrix: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1e-6
) -> np.ndarray:
    """
    Iterative Proportional Fitting (IPF) algorithm.
    
    Adjusts a matrix to match target row and column sums while preserving
    the structure of the seed matrix as much as possible.
    
    Parameters:
    -----------
    seed_matrix : np.ndarray
        Initial matrix to adjust
    row_targets : np.ndarray
        Target row sums
    col_targets : np.ndarray
        Target column sums
    max_iterations : int
        Maximum iterations for convergence
    tolerance : float
        Convergence tolerance
        
    Returns:
    --------
    np.ndarray
        Adjusted matrix matching marginal totals
    """
    matrix = seed_matrix.copy().astype(float)
    
    # Handle zero targets to avoid division by zero
    row_targets = np.where(row_targets == 0, 1e-10, row_targets)
    col_targets = np.where(col_targets == 0, 1e-10, col_targets)
    
    for iteration in range(max_iterations):
        prev_matrix = matrix.copy()
        
        # Row adjustment
        row_sums = matrix.sum(axis=1)
        row_sums = np.where(row_sums == 0, 1e-10, row_sums)
        row_factors = row_targets / row_sums
        matrix = matrix * row_factors[:, np.newaxis]
        
        # Column adjustment
        col_sums = matrix.sum(axis=0)
        col_sums = np.where(col_sums == 0, 1e-10, col_sums)
        col_factors = col_targets / col_sums
        matrix = matrix * col_factors[np.newaxis, :]
        
        # Check convergence
        if np.allclose(matrix, prev_matrix, rtol=tolerance, atol=tolerance):
            break
    
    return matrix


def configuration_model(
    flow_matrix: np.ndarray,
    n_randomizations: int = 100,
    preserve_structure: bool = True
) -> List[np.ndarray]:
    """
    Configuration model for randomizing flows preserving marginal totals.
    
    Implements a configuration model that preserves the degree sequence
    (marginal totals) while randomizing the internal structure.
    
    Parameters:
    -----------
    flow_matrix : np.ndarray
        Original flow matrix (n x n)
    n_randomizations : int
        Number of randomized matrices to generate
    preserve_structure : bool
        If True, uses IPF to preserve structure; if False, uses edge swapping
        
    Returns:
    --------
    List[np.ndarray]
        List of randomized flow matrices
    """
    n = flow_matrix.shape[0]
    row_sums = flow_matrix.sum(axis=1)
    col_sums = flow_matrix.sum(axis=0)
    
    randomized_matrices = []
    
    if preserve_structure:
        for _ in range(n_randomizations):
            seed = np.random.rand(n, n)
            seed[flow_matrix == 0] = 0
            
            randomized = iterative_proportional_fitting(seed, row_sums, col_sums)
            randomized_matrices.append(randomized)
    else:
        for _ in range(n_randomizations):
            randomized = edge_swap_randomization(flow_matrix, n_swaps=1000)
            randomized_matrices.append(randomized)
    
    return randomized_matrices


def edge_swap_randomization(
    flow_matrix: np.ndarray,
    n_swaps: int = 1000
) -> np.ndarray:
    """
    Edge swap randomization for binary or weighted networks.
    
    Preserves row and column degrees (for binary) or approximately
    preserves marginals (for weighted).
    
    Parameters:
    -----------
    flow_matrix : np.ndarray
        Original flow matrix
    n_swaps : int
        Number of edge swaps to attempt
        
    Returns:
    --------
    np.ndarray
        Randomized matrix
    """
    matrix = flow_matrix.copy()
    n = matrix.shape[0]
    
    nonzero_indices = np.argwhere(matrix > 0)
    
    if len(nonzero_indices) < 4:
        return matrix
    
    swaps_attempted = 0
    swaps_made = 0
    
    while swaps_made < n_swaps and swaps_attempted < n_swaps * 10:
        swaps_attempted += 1
        
        idx1, idx2 = np.random.choice(len(nonzero_indices), 2, replace=False)
        i, j = nonzero_indices[idx1]
        k, l = nonzero_indices[idx2]
        
        if i != l and k != j and matrix[i, l] == 0 and matrix[k, j] == 0:
            w1, w2 = matrix[i, j], matrix[k, l]
            matrix[i, j] = 0
            matrix[k, l] = 0
            matrix[i, l] = w1
            matrix[k, j] = w2
            
            nonzero_indices[idx1] = [i, l]
            nonzero_indices[idx2] = [k, j]
            
            swaps_made += 1
    
    return matrix


def randomize_flows_preserving_marginals(
    flow_matrix: np.ndarray,
    n_permutations: int = 1000,
    method: str = 'ipf',
    random_state: Optional[int] = None
) -> Dict:
    """
    Null Model 1: Equilibrium (Randomized Flows)
    
    Randomizes migration flows while preserving marginal totals
    (origin and destination sums). Tests if observed entropy production
    is significantly different from equilibrium/randomized flows.
    
    Parameters:
    -----------
    flow_matrix : np.ndarray
        Original migration flow matrix (n x n)
    n_permutations : int
        Number of randomizations to generate
    method : str
        'ipf' for iterative proportional fitting, 'edge_swap' for edge swapping
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    Dict
        Dictionary containing:
        - 'randomized_matrices': List of randomized matrices
        - 'observed_marginals': (row_sums, col_sums) tuple
        - 'method': Method used
        - 'n_permutations': Number of permutations
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Validate input
    if flow_matrix.ndim != 2 or flow_matrix.shape[0] != flow_matrix.shape[1]:
        raise ValueError("Flow matrix must be square (n x n)")
    
    # Compute observed marginals
    row_sums = flow_matrix.sum(axis=1)
    col_sums = flow_matrix.sum(axis=0)
    
    # Generate randomized matrices
    if method == 'ipf':
        randomized = configuration_model(
            flow_matrix, 
            n_randomizations=n_permutations,
            preserve_structure=True
        )
    elif method == 'edge_swap':
        randomized = configuration_model(
            flow_matrix,
            n_randomizations=n_permutations,
            preserve_structure=False
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'randomized_matrices': randomized,
        'observed_marginals': (row_sums, col_sums),
        'method': method,
        'n_permutations': n_permutations
    }


def gravity_model_predictions(
    distances: np.ndarray,
    masses: np.ndarray,
    params: Dict[str, float]
) -> np.ndarray:
    """
    Null Model 2: Generate gravity model predictions.
    
    Traditional gravity model: J_ij = k * (M_i^α * M_j^β) / d_ij^γ
    
    Parameters:
    -----------
    distances : np.ndarray
        Distance matrix (n x n)
    masses : np.ndarray
        Mass/population vector (n,)
    params : Dict[str, float]
        Gravity model parameters with keys:
        - 'alpha': Origin mass exponent
        - 'beta': Destination mass exponent  
        - 'gamma': Distance decay exponent
        - 'k': Scaling constant
        
    Returns:
    --------
    np.ndarray
        Predicted flow matrix (n x n)
    """
    n = len(masses)
    
    # Extract parameters with defaults
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 1.0)
    gamma = params.get('gamma', 1.0)
    k = params.get('k', 1.0)
    
    # Avoid division by zero
    distances = distances.copy()
    np.fill_diagonal(distances, np.inf)
    
    # Gravity model: T_ij = k * (M_i^α * M_j^β) / d_ij^γ
    mass_i = masses[:, np.newaxis] ** alpha
    mass_j = masses[np.newaxis, :] ** beta
    distance_factor = distances ** (-gamma)
    
    predicted_flows = k * mass_i * mass_j * distance_factor
    
    # Set diagonal to zero (no self-flows)
    np.fill_diagonal(predicted_flows, 0)
    
    return predicted_flows


def fit_gravity_model(
    observed_flows: np.ndarray,
    distances: np.ndarray,
    masses: np.ndarray,
    param_grid: Optional[Dict] = None
) -> Dict:
    """
    Fit gravity model parameters to observed flows.
    
    Parameters:
    -----------
    observed_flows : np.ndarray
        Observed flow matrix (n x n)
    distances : np.ndarray
        Distance matrix (n x n)
    masses : np.ndarray
        Mass/population vector (n,)
    param_grid : dict, optional
        Grid of parameters to search
        
    Returns:
    --------
    Dict
        Best parameters and fit statistics:
        - 'best_params': Best parameter dict
        - 'best_rmse': Best RMSE
        - 'r2': R-squared of best fit
        - 'all_results': All grid search results
    """
    if param_grid is None:
        param_grid = {
            'alpha': [0.5, 1.0, 1.5],
            'beta': [0.5, 1.0, 1.5],
            'gamma': [0.5, 1.0, 1.5, 2.0],
            'k': [1.0]
        }
    
    best_rmse = np.inf
    best_params = {}
    best_r2 = -np.inf
    all_results = []
    
    total_observed = observed_flows.sum()
    
    for alpha in param_grid['alpha']:
        for beta in param_grid['beta']:
            for gamma in param_grid['gamma']:
                for k in param_grid['k']:
                    params = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'k': k}
                    
                    predicted = gravity_model_predictions(
                        distances, masses, params
                    )
                    
                    # Normalize predicted to match total observed
                    if predicted.sum() > 0:
                        predicted = predicted * (total_observed / predicted.sum())
                    
                    # Compute metrics (only for non-zero observed)
                    mask = observed_flows > 0
                    if mask.sum() > 0:
                        rmse = np.sqrt(np.mean((observed_flows[mask] - predicted[mask])**2))
                        
                        # R-squared
                        ss_res = np.sum((observed_flows[mask] - predicted[mask])**2)
                        ss_tot = np.sum((observed_flows[mask] - observed_flows[mask].mean())**2)
                        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    else:
                        rmse = np.inf
                        r2 = -np.inf
                    
                    all_results.append({
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'k': k,
                        'rmse': rmse,
                        'r2': r2
                    })
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = params.copy()
                        best_r2 = r2
    
    return {
        'best_params': best_params,
        'best_rmse': best_rmse,
        'r2': best_r2,
        'all_results': all_results
    }


def compare_to_null_models(
    observed_metrics: Dict[str, float],
    null_distribution: Dict,
    test_type: str = 'greater'
) -> Dict:
    """
    Test observed metrics against null distribution.
    
    Parameters:
    -----------
    observed_metrics : dict
        Dictionary of observed metric values {name: value}
    null_distribution : dict
        Output from randomized null model
    test_type : str
        'greater' (observed > null), 'less', or 'two-sided'
        
    Returns:
    --------
    Dict
        Test results with p-values, z-scores, and conclusions
    """
    results = {}
    
    null_values = null_distribution.get('values', null_distribution.get('null_values', []))
    null_values = np.array(null_values)
    
    if len(null_values) == 0:
        raise ValueError("No null values provided in null_distribution")
    
    null_mean = np.mean(null_values)
    null_std = np.std(null_values)
    
    for metric_name, observed in observed_metrics.items():
        # Compute z-score
        if null_std > 0:
            z_score = (observed - null_mean) / null_std
        else:
            z_score = np.inf if observed > null_mean else -np.inf
        
        # Compute p-value based on test type
        if test_type == 'greater':
            p_value = np.mean(null_values >= observed)
        elif test_type == 'less':
            p_value = np.mean(null_values <= observed)
        else:  # two-sided
            p_value = np.mean(np.abs(null_values - null_mean) >= np.abs(observed - null_mean))
        
        # Compute effect size (Cohen's d)
        if null_std > 0:
            cohens_d = (observed - null_mean) / null_std
        else:
            cohens_d = np.nan
        
        # Determine significance
        is_significant = p_value < 0.05
        
        results[metric_name] = {
            'observed': observed,
            'null_mean': null_mean,
            'null_std': null_std,
            'z_score': z_score,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'is_significant': is_significant,
            'effect_size': interpret_effect_size(abs(cohens_d)),
            'conclusion': 'Reject null' if is_significant else 'Fail to reject null'
        }
    
    return results


def compute_entropy_production(
    flow_matrix: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute entropy production (σ) from flow matrix.
    
    σ = Σ_ij F_ij * ln(F_ij / F_ji)
    
    Parameters:
    -----------
    flow_matrix : np.ndarray
        Flow matrix
    epsilon : float
        Small value to avoid log(0)
        
    Returns:
    --------
    float
        Entropy production σ
    """
    F = flow_matrix + epsilon
    F_rev = flow_matrix.T + epsilon
    
    sigma = np.sum(F * np.log(F / F_rev))
    
    return sigma


def interpret_effect_size(cohens_d: float) -> str:
    """Interpret Cohen's d effect size."""
    if np.isnan(cohens_d):
        return 'undefined'
    elif cohens_d < 0.2:
        return 'negligible'
    elif cohens_d < 0.5:
        return 'small'
    elif cohens_d < 0.8:
        return 'medium'
    else:
        return 'large'


if __name__ == '__main__':
    print("Null Models Module - Example Usage")
    print("=" * 50)
    
    # Generate synthetic data
    n = 20
    np.random.seed(42)
    synthetic_flows = np.random.lognormal(0, 1, size=(n, n))
    np.fill_diagonal(synthetic_flows, 0)
    
    print(f"\nSynthetic flow matrix: {n}x{n}")
    print(f"Total flow: {synthetic_flows.sum():.1f}")
    print(f"Non-zero entries: {np.count_nonzero(synthetic_flows)}")
    
    # Test Null Model 1: Equilibrium
    print("\n" + "=" * 50)
    print("NULL MODEL 1: Equilibrium (Randomized Flows)")
    print("=" * 50)
    
    null_results = randomize_flows_preserving_marginals(
        synthetic_flows,
        n_permutations=100,
        method='ipf',
        random_state=42
    )
    
    # Compute entropy production for null models
    null_entropies = [
        compute_entropy_production(m) for m in null_results['randomized_matrices']
    ]
    null_entropies = np.array(null_entropies)
    
    observed_entropy = compute_entropy_production(synthetic_flows)
    
    print(f"\nObserved entropy production: {observed_entropy:.4f}")
    print(f"Null distribution mean: {null_entropies.mean():.4f} ± {null_entropies.std():.4f}")
    print(f"95% CI: [{np.percentile(null_entropies, 2.5):.4f}, {np.percentile(null_entropies, 97.5):.4f}]")
    
    # Test Null Model 2: Gravity
    print("\n" + "=" * 50)
    print("NULL MODEL 2: Substantialist (Gravity Model)")
    print("=" * 50)
    
    populations = np.random.lognormal(mean=10, sigma=1, size=n)
    coords = np.random.rand(n, 2) * 1000
    distances = squareform(pdist(coords, metric='euclidean'))
    
    fit_results = fit_gravity_model(
        synthetic_flows, distances, populations
    )
    
    print(f"\nBest fit parameters: {fit_results['best_params']}")
    print(f"Best RMSE: {fit_results['best_rmse']:.4f}")
    print(f"R-squared: {fit_results['r2']:.4f}")
