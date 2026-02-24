"""
Entropy Production Density Calculations

This module implements the core thermodynamic formalism for calculating
entropy production in urban systems based on population flows.

Mathematical Formalism:
----------------------
For county/MSA i at time t:
    σ_i(t) = (1/2) * Σ_{j≠i} J_ij(t) * X_ij(t)

where:
    - J_ij = migration flow from i to j (people/year)
    - X_ij = thermodynamic affinity = ln(p_i/p_j) - ln(p_i^eq/p_j^eq)
    - p_i = N_i / ΣN_k (population share)
    - p_i^eq = equilibrium population share (gravity model or long-term average)

The factor 1/2 accounts for double-counting in symmetric summation.

References:
- Prigogine, I. (1967). Introduction to Thermodynamics of Irreversible Processes
- Kondepudi, D. & Prigogine, I. (1998). Modern Thermodynamics
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
import warnings

try:
    from scipy.sparse import csr_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    csr_matrix = None


def compute_thermodynamic_affinity(
    p_i: Union[float, np.ndarray],
    p_j: Union[float, np.ndarray],
    p_eq_i: Union[float, np.ndarray],
    p_eq_j: Union[float, np.ndarray],
    epsilon: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    Compute thermodynamic affinity X_ij between regions i and j.
    
    The affinity measures the deviation from equilibrium and drives
    the irreversible flows in the system.
    
    Formula:
        X_ij = ln(p_i/p_j) - ln(p_i^eq/p_j^eq)
             = ln(p_i/p_i^eq) - ln(p_j/p_j^eq)
    
    Interpretation:
        - X_ij > 0: Flow expected from j to i (i is "hotter" thermodynamically)
        - X_ij < 0: Flow expected from i to j (j is "hotter" thermodynamically)  
        - X_ij = 0: Equilibrium condition (no net flow)
    
    Parameters:
    -----------
    p_i : float or np.ndarray
        Actual population share(s) of region i
    p_j : float or np.ndarray
        Actual population share(s) of region j
    p_eq_i : float or np.ndarray
        Equilibrium population share(s) of region i
    p_eq_j : float or np.ndarray
        Equilibrium population share(s) of region j
    epsilon : float, default=1e-10
        Small constant to avoid log(0)
    
    Returns:
    --------
    float or np.ndarray
        Thermodynamic affinity X_ij (dimensionless)
    
    Example:
    --------
    >>> # Single pair calculation
    >>> X = compute_thermodynamic_affinity(0.3, 0.1, 0.25, 0.15)
    >>> print(f"Affinity: {X:.4f}")
    
    >>> # Vectorized calculation for multiple regions
    >>> p = np.array([0.3, 0.2, 0.15, 0.1])
    >>> p_eq = np.array([0.25, 0.22, 0.18, 0.12])
    >>> X_matrix = compute_thermodynamic_affinity(
    ...     p[:, None], p[None, :], p_eq[:, None], p_eq[None, :]
    ... )
    """
    # Clip to avoid log(0)
    p_i = np.maximum(np.asarray(p_i, dtype=float), epsilon)
    p_j = np.maximum(np.asarray(p_j, dtype=float), epsilon)
    p_eq_i = np.maximum(np.asarray(p_eq_i, dtype=float), epsilon)
    p_eq_j = np.maximum(np.asarray(p_eq_j, dtype=float), epsilon)
    
    # X_ij = ln(p_i/p_j) - ln(p_i^eq/p_j^eq)
    #      = ln(p_i) - ln(p_j) - ln(p_i^eq) + ln(p_eq_j)
    #      = ln(p_i/p_i^eq) - ln(p_j/p_eq_j)
    
    log_ratio_actual = np.log(p_i / p_j)
    log_ratio_equilibrium = np.log(p_eq_i / p_eq_j)
    
    X_ij = log_ratio_actual - log_ratio_equilibrium
    
    return X_ij


def equilibrium_gravity_model(
    distances: np.ndarray,
    populations: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    k: float = 1.0
) -> np.ndarray:
    """
    Compute equilibrium population distribution using gravity model.
    
    The gravity model provides a null hypothesis for equilibrium flows
    based on spatial interaction theory.
    
    Formula:
        p_i^eq ∝ (N_i)^α * Σ_j (N_j^β / d_ij^k)
    
    where:
        - N_i, N_j = populations
        - d_ij = distance between regions i and j
        - α, β = production/attraction exponents (typically ~1.0)
        - k = distance decay parameter (typically 1.0-2.0)
    
    Parameters:
    -----------
    distances : np.ndarray, shape (n, n)
        Distance matrix between regions (symmetric, diagonal is 0 or inf)
    populations : np.ndarray, shape (n,)
        Population vector for each region
    alpha : float, default=1.0
        Exponent for origin population effect
    beta : float, default=1.0
        Exponent for destination population effect  
    k : float, default=1.0
        Distance decay exponent
    
    Returns:
    --------
    np.ndarray, shape (n,)
        Equilibrium population shares p_eq
    
    Notes:
    ------
    - Distance matrix should have np.inf or very large values on diagonal
      to prevent self-interaction
    - Typical values: alpha=1.0, beta=1.0, k=1.0 to 2.0
    """
    n = len(populations)
    populations = np.asarray(populations, dtype=float)
    distances = np.asarray(distances, dtype=float)
    
    # Ensure diagonal doesn't cause division by zero
    distances = distances.copy()
    np.fill_diagonal(distances, np.inf)
    
    # Gravity model potential
    # T_ij ∝ (N_i)^α * (N_j)^β / d_ij^k
    with np.errstate(divide='ignore', invalid='ignore'):
        distance_factor = np.where(
            distances > 0,
            distances ** (-k),
            0
        )
    
    # Origin potential: sum over all destinations
    origin_potential = (populations ** beta) @ distance_factor.T
    
    # Equilibrium weight (origin effect * destination accessibility)
    equilibrium_weights = (populations ** alpha) * origin_potential
    
    # Normalize to get shares
    p_eq = equilibrium_weights / equilibrium_weights.sum()
    
    return p_eq


def compute_entropy_production(
    flow_matrix: np.ndarray,
    populations: np.ndarray,
    equilibrium: Optional[np.ndarray] = None,
    return_components: bool = False
) -> Union[float, Tuple[float, dict]]:
    """
    Calculate entropy production density σ_i for each region.
    
    Core thermodynamic quantity measuring the rate of entropy generation
    due to population flows in urban systems.
    
    Formula:
        σ_i = (1/2) * Σ_{j≠i} J_ij * X_ij
    
    where:
        - J_ij = flow from i to j (symmetric: J_ij = -J_ji for net flows)
        - X_ij = thermodynamic affinity
    
    Parameters:
    -----------
    flow_matrix : np.ndarray, shape (n, n)
        Migration flow matrix where flow_matrix[i,j] = flow from i to j.
        For net flows: flow_matrix[i,j] = -flow_matrix[j,i]
        For gross flows: flow_matrix[i,j] >= 0
    populations : np.ndarray, shape (n,)
        Current population of each region
    equilibrium : np.ndarray, optional, shape (n,)
        Equilibrium population shares. If None, uses long-term average
        of input populations (p_i = p_i^eq, so all affinities are zero).
    return_components : bool, default=False
        If True, also return detailed components (flows, affinities, contributions)
    
    Returns:
    --------
    float or tuple
        If return_components=False: Total entropy production σ (scalar)
        If return_components=True: (σ, components_dict) where components_dict
        contains:
            - 'sigma_per_region': σ_i for each region
            - 'flow_matrix': input flow matrix
            - 'affinities': X_ij matrix
            - 'entropy_contributions': J_ij * X_ij for each pair
            - 'populations': population shares used
            - 'equilibrium': equilibrium shares used
    
    Example:
    --------
    >>> # Simple 3-region example
    >>> flows = np.array([
    ...     [0, 100, 50],
    ...     [80, 0, 30],
    ...     [40, 20, 0]
    ... ])
    >>> pops = np.array([1000, 800, 600])
    >>> sigma = compute_entropy_production(flows, pops)
    >>> print(f"Total entropy production: {sigma:.2f} people/year")
    
    >>> # With equilibrium model
    >>> distances = np.array([[0, 100, 200], [100, 0, 150], [200, 150, 0]])
    >>> p_eq = equilibrium_gravity_model(distances, pops)
    >>> sigma, details = compute_entropy_production(
    ...     flows, pops, equilibrium=p_eq, return_components=True
    ... )
    """
    flow_matrix = np.asarray(flow_matrix, dtype=float)
    populations = np.asarray(populations, dtype=float)
    n = len(populations)
    
    if flow_matrix.shape != (n, n):
        raise ValueError(
            f"Flow matrix shape {flow_matrix.shape} incompatible with "
            f"population vector length {n}"
        )
    
    # Compute population shares
    p = populations / populations.sum()
    
    # Compute equilibrium shares
    if equilibrium is None:
        # Default: uniform equilibrium (all affinities measure deviation from uniformity)
        p_eq = np.ones(n) / n
        warnings.warn(
            "No equilibrium specified. Using uniform distribution. "
            "Consider providing a gravity model equilibrium for better results."
        )
    else:
        p_eq = np.asarray(equilibrium, dtype=float)
        if len(p_eq) != n:
            raise ValueError(
                f"Equilibrium vector length {len(p_eq)} doesn't match "
                f"population vector length {n}"
            )
        # Renormalize to ensure it's a valid probability distribution
        p_eq = p_eq / p_eq.sum()
    
    # Compute affinity matrix X_ij
    # X_ij = ln(p_i/p_j) - ln(p_i^eq/p_j^eq)
    p_i = p[:, np.newaxis]  # Column vector
    p_j = p[np.newaxis, :]  # Row vector
    p_eq_i = p_eq[:, np.newaxis]
    p_eq_j = p_eq[np.newaxis, :]
    
    X = compute_thermodynamic_affinity(p_i, p_j, p_eq_i, p_eq_j)
    
    # For net flows, use antisymmetric part
    # J_ij^net = (flow_ij - flow_ji) / 2
    J_net = (flow_matrix - flow_matrix.T) / 2.0
    
    # Entropy production contributions: J_ij * X_ij
    # The 1/2 factor in the formula accounts for double counting
    contributions = J_net * X
    
    # Entropy production per region: σ_i = Σ_j J_ij * X_ij
    # (Note: we don't use 1/2 here because we're not summing over all pairs twice)
    sigma_per_region = np.sum(contributions, axis=1)
    
    # Total entropy production: σ = (1/2) * Σ_i Σ_j J_ij * X_ij
    # This is equivalent to sum of upper triangle since J*X is antisymmetric
    sigma_total = np.sum(np.triu(contributions, k=1)) * 2
    # Alternative: sigma_total = sigma_per_region.sum() / 2
    
    if return_components:
        components = {
            'sigma_per_region': sigma_per_region,
            'sigma_total': sigma_total,
            'flow_matrix': flow_matrix,
            'net_flows': J_net,
            'affinities': X,
            'entropy_contributions': contributions,
            'populations': populations,
            'population_shares': p,
            'equilibrium_shares': p_eq,
        }
        return sigma_total, components
    
    return sigma_total


def compute_total_entropy_production(
    msa_flows: dict,
    msa_pops: dict,
    equilibrium_model: str = 'gravity',
    distance_matrix: Optional[np.ndarray] = None,
    return_by_msa: bool = False
) -> Union[float, pd.DataFrame]:
    """
    Compute system-wide entropy production for multiple MSAs over time.
    
    This function aggregates entropy production across all MSAs and time periods,
    providing a comprehensive measure of the system's dissipative activity.
    
    Parameters:
    -----------
    msa_flows : dict
        Dictionary mapping year -> flow matrix for that year.
        Each flow matrix has shape (n_msa, n_msa).
    msa_pops : dict
        Dictionary mapping year -> population vector for that year.
        Each vector has length n_msa.
    equilibrium_model : str, default='gravity'
        Method for computing equilibrium:
        - 'gravity': Use gravity model with distances
        - 'average': Use long-term average populations
        - 'first_year': Use first year as equilibrium reference
    distance_matrix : np.ndarray, optional
        Required if equilibrium_model='gravity'. Shape (n_msa, n_msa).
    return_by_msa : bool, default=False
        If True, return DataFrame with entropy production by MSA and year.
        If False, return scalar total entropy production.
    
    Returns:
    --------
    float or pd.DataFrame
        If return_by_msa=False: Total entropy production across all years
        If return_by_msa=True: DataFrame with columns ['year', 'msa_id', 
        'entropy_production', 'population', 'sigma_per_capita']
    
    Example:
    --------
    >>> # Multi-year analysis
    >>> years = list(range(2000, 2024))
    >>> flows_by_year = {y: np.random.rand(100, 100) * 1000 for y in years}
    >>> pops_by_year = {y: np.random.rand(100) * 1000000 for y in years}
    >>> 
    >>> sigma_by_msa = compute_total_entropy_production(
    ...     flows_by_year,
    ...     pops_by_year,
    ...     equilibrium_model='average',
    ...     return_by_msa=True
    ... )
    """
    years = sorted(msa_flows.keys())
    n_msas = len(msa_pops[years[0]])
    
    # Compute equilibrium based on model
    if equilibrium_model == 'gravity':
        if distance_matrix is None:
            raise ValueError("distance_matrix required for gravity equilibrium model")
        avg_pops = np.mean([msa_pops[y] for y in years], axis=0)
        equilibrium = equilibrium_gravity_model(distance_matrix, avg_pops)
    elif equilibrium_model == 'average':
        avg_pops = np.mean([msa_pops[y] for y in years], axis=0)
        equilibrium = avg_pops / avg_pops.sum()
    elif equilibrium_model == 'first_year':
        first_year_pops = msa_pops[years[0]]
        equilibrium = first_year_pops / first_year_pops.sum()
    else:
        raise ValueError(f"Unknown equilibrium_model: {equilibrium_model}")
    
    results = []
    total_sigma = 0.0
    
    for year in years:
        flows = msa_flows[year]
        pops = msa_pops[year]
        
        sigma, components = compute_entropy_production(
            flows, pops, equilibrium, return_components=True
        )
        
        total_sigma += sigma
        
        if return_by_msa:
            sigma_per_region = components['sigma_per_region']
            for msa_idx in range(n_msas):
                results.append({
                    'year': year,
                    'msa_id': msa_idx,
                    'entropy_production': sigma_per_region[msa_idx],
                    'population': pops[msa_idx],
                    'sigma_per_capita': sigma_per_region[msa_idx] / pops[msa_idx] if pops[msa_idx] > 0 else 0,
                    'population_share': components['population_shares'][msa_idx],
                    'equilibrium_share': components['equilibrium_shares'][msa_idx],
                })
    
    if return_by_msa:
        return pd.DataFrame(results)
    else:
        return total_sigma


def decompose_entropy_production(
    flow_matrix: np.ndarray,
    populations: np.ndarray,
    equilibrium: Optional[np.ndarray] = None
) -> dict:
    """
    Decompose entropy production into contributions from different scales.
    
    Breaks down total entropy production into:
    1. Spatial component: due to geographic distribution
    2. Temporal component: due to deviation from equilibrium
    3. Flow intensity: magnitude of migration flows
    
    Parameters:
    -----------
    flow_matrix : np.ndarray, shape (n, n)
        Migration flow matrix
    populations : np.ndarray, shape (n,)
        Population vector
    equilibrium : np.ndarray, optional
        Equilibrium population shares
    
    Returns:
    --------
    dict
        Decomposition components:
        - 'total_sigma': Total entropy production
        - 'spatial_contribution': Contribution from spatial structure
        - 'temporal_contribution': Contribution from non-equilibrium dynamics
        - 'flow_magnitude': Overall flow intensity
        - 'efficiency': sigma / (flow_magnitude * max_possible_affinity)
    """
    sigma, comps = compute_entropy_production(
        flow_matrix, populations, equilibrium, return_components=True
    )
    
    # Flow magnitude (total absolute flow)
    flow_magnitude = np.abs(comps['net_flows']).sum() / 2
    
    # Average affinity
    mask = comps['affinities'] != 0
    if mask.any():
        avg_affinity = np.abs(comps['affinities'][mask]).mean()
        max_affinity = np.abs(comps['affinities'][mask]).max()
    else:
        avg_affinity = 0
        max_affinity = 1  # Avoid division by zero
    
    # Efficiency: how effectively flows are coupled to affinities
    # σ = Σ J_ij * X_ij, so efficiency = σ / (|J|_total * |X|_max)
    efficiency = sigma / (flow_magnitude * max_affinity) if flow_magnitude > 0 else 0
    
    # Decomposition
    # Spatial contribution: related to population share deviations
    p = comps['population_shares']
    p_eq = comps['equilibrium_shares']
    spatial_divergence = np.sum(np.abs(p - p_eq))
    
    decomposition = {
        'total_sigma': sigma,
        'flow_magnitude': flow_magnitude,
        'average_affinity': avg_affinity,
        'max_affinity': max_affinity,
        'efficiency': efficiency,
        'spatial_divergence': spatial_divergence,
        'sigma_per_unit_flow': sigma / flow_magnitude if flow_magnitude > 0 else 0,
    }
    
    return decomposition
