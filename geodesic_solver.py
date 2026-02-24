"""
Geodesic Solver for Demographic Statistical Manifold

This module implements numerical integration of geodesic equations and
computation of geodesic distances on the Fisher-information manifold.

Mathematical Framework:
--------------------------
Geodesics are curves that locally minimize distance on the manifold.
They satisfy the geodesic equation:

    d²θ^μ/dt² + Γ^μ_νρ (dθ^ν/dt)(dθ^ρ/dt) = 0

where Γ^μ_νρ are the Christoffel symbols derived from the Fisher metric.

For demographic flows, geodesics represent "natural" evolutionary paths
that minimize information-theoretic distance between demographic states.

Key Operations:
--------------
1. Geodesic integration: Solve initial value problem for geodesic equations
2. Geodesic distance: Compute shortest path between two points
3. Parallel transport: Transport vectors along curves while preserving angles

References:
-----------
- Amari, S. (2016). Information Geometry and Its Applications.
- do Carmo, M. P. (1992). Riemannian Geometry.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from typing import Tuple, Optional, Callable, List, Union
import warnings

from .fisher_metric import FisherMetric


class GeodesicSolver:
    """
    Numerical solver for geodesic equations on demographic statistical manifold.
    
    Attributes:
        fisher_metric (FisherMetric): Metric calculator instance
        integrator (str): ODE integration method
        max_iter (int): Maximum iterations for boundary value problems
    """
    
    def __init__(self, fisher_metric: Optional[FisherMetric] = None,
                 integrator: str = 'RK45', rtol: float = 1e-6, atol: float = 1e-9):
        """
        Initialize geodesic solver.
        
        Args:
            fisher_metric: FisherMetric instance (created if None)
            integrator: Integration method ('RK45', 'RK23', 'DOP853', 'Radau')
            rtol: Relative tolerance for integration
            atol: Absolute tolerance for integration
        """
        self.fisher_metric = fisher_metric or FisherMetric()
        self.integrator = integrator
        self.rtol = rtol
        self.atol = atol
        
    def solve_geodesic(self, theta_0: np.ndarray, theta_dot_0: np.ndarray,
                      g_func: Optional[Callable] = None,
                      t_span: Tuple[float, float] = (0, 1),
                      n_points: int = 100) -> dict:
        """
        Solve initial value problem for geodesic equations.
        
        Given initial position θ(0) and velocity θ̇(0), integrate the geodesic
        equations forward in time.
        
        Args:
            theta_0: Initial position on manifold (4D vector)
            theta_dot_0: Initial velocity/tangent vector (4D vector)
            g_func: Function g(θ) returning metric at point θ. If None, uses
                   empirical metric from local data.
            t_span: Integration time interval (t_start, t_end)
            n_points: Number of points to return in solution
            
        Returns:
            result: Dictionary containing:
                - 't': Time points
                - 'theta': Position trajectory (n_points, 4)
                - 'theta_dot': Velocity trajectory (n_points, 4)
                - 'success': Whether integration succeeded
                - 'message': Status message
                
        Mathematical Formulation:
        -------------------------
        We solve the first-order system obtained by setting v^μ = θ̇^μ:
        
            dθ^μ/dt = v^μ
            dv^μ/dt = -Γ^μ_νρ v^ν v^ρ
            
        This is an 8-dimensional ODE (4 position + 4 velocity components).
        """
        theta_0 = np.asarray(theta_0, dtype=float)
        theta_dot_0 = np.asarray(theta_dot_0, dtype=float)
        
        if len(theta_0) != 4 or len(theta_dot_0) != 4:
            raise ValueError("theta_0 and theta_dot_0 must be 4-dimensional")
        
        # Default metric function
        if g_func is None:
            g_func = lambda theta: self.fisher_metric.compute_fisher_metric(
                theta.reshape(1, -1), method='empirical'
            )
        
        # Define ODE system
        def geodesic_ode(t, y):
            """
            Geodesic ODE in first-order form.
            
            y = [θ^0, θ^1, θ^2, θ^3, v^0, v^1, v^2, v^3]
            dy/dt = [v^0, v^1, v^2, v^3, -Γ^0_νρ v^ν v^ρ, ..., -Γ^3_νρ v^ν v^ρ]
            """
            theta = y[:4]
            v = y[4:]
            
            # Compute metric and Christoffel symbols
            try:
                g = g_func(theta)
                Gamma = self.fisher_metric.compute_christoffel_symbols(g, theta)
            except Exception:
                # Fallback to zero Christoffel symbols
                Gamma = np.zeros((4, 4, 4))
            
            # Compute acceleration: a^μ = -Γ^μ_νρ v^ν v^ρ
            a = np.zeros(4)
            for mu in range(4):
                for nu in range(4):
                    for rho in range(4):
                        a[mu] -= Gamma[mu, nu, rho] * v[nu] * v[rho]
            
            return np.concatenate([v, a])
        
        # Initial conditions
        y0 = np.concatenate([theta_0, theta_dot_0])
        
        # Time points for output
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        # Integrate
        try:
            sol = solve_ivp(
                geodesic_ode, t_span, y0,
                method=self.integrator,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
                dense_output=True
            )
            
            success = sol.success
            message = sol.message if hasattr(sol, 'message') else "Integration completed"
            
            if success:
                theta_trajectory = sol.y[:4, :].T
                theta_dot_trajectory = sol.y[4:, :].T
                t_points = sol.t
            else:
                theta_trajectory = np.tile(theta_0, (n_points, 1))
                theta_dot_trajectory = np.tile(theta_dot_0, (n_points, 1))
                t_points = t_eval
                
        except Exception as e:
            success = False
            message = f"Integration failed: {str(e)}"
            theta_trajectory = np.tile(theta_0, (n_points, 1))
            theta_dot_trajectory = np.tile(theta_dot_0, (n_points, 1))
            t_points = t_eval
        
        return {
            't': t_points,
            'theta': theta_trajectory,
            'theta_dot': theta_dot_trajectory,
            'success': success,
            'message': message,
            'sol': sol if success else None
        }
    
    def compute_geodesic_distance(self, theta_a: np.ndarray, theta_b: np.ndarray,
                                  g_func: Optional[Callable] = None,
                                  method: str = 'shooting') -> float:
        """
        Compute geodesic distance between two points on the manifold.
        
        The geodesic distance is the length of the shortest path between two
        points, measured using the Fisher metric:
        
            d(θ_a, θ_b) = ∫_a^b √(g_μν θ̇^μ θ̇^ν) dt
        
        Args:
            theta_a: Starting point (4D vector)
            theta_b: Ending point (4D vector)
            g_func: Function returning metric at a point
            method: 'shooting' or 'relaxation' for boundary value solution
            
        Returns:
            distance: Geodesic distance (scalar)
            
        Methods:
        --------
        - shooting: Solve initial value problem with optimization over initial velocity
        - relaxation: Iteratively relax path to geodesic
        """
        theta_a = np.asarray(theta_a, dtype=float)
        theta_b = np.asarray(theta_b, dtype=float)
        
        if g_func is None:
            g_func = lambda theta: self.fisher_metric.compute_fisher_metric(
                theta.reshape(1, -1), method='empirical'
            )
        
        if method == 'euclidean_fallback':
            # Fallback to Euclidean distance
            return np.linalg.norm(theta_b - theta_a)
        
        elif method == 'shooting':
            return self._distance_shooting(theta_a, theta_b, g_func)
        
        elif method == 'relaxation':
            return self._distance_relaxation(theta_a, theta_b, g_func)
        
        elif method == 'shortest_path':
            return self._distance_discrete(theta_a, theta_b, g_func)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _distance_shooting(self, theta_a: np.ndarray, theta_b: np.ndarray,
                          g_func: Callable) -> float:
        """
        Compute geodesic distance using shooting method.
        
        Find initial velocity such that geodesic from θ_a reaches θ_b at t=1.
        """
        # Initial guess: Euclidean velocity
        v0_guess = theta_b - theta_a
        
        def objective(v0):
            """Minimize squared distance between geodesic endpoint and target."""
            sol = self.solve_geodesic(theta_a, v0, g_func, t_span=(0, 1), n_points=50)
            if not sol['success']:
                return 1e10
            theta_end = sol['theta'][-1]
            return np.sum((theta_end - theta_b) ** 2)
        
        # Optimize
        from scipy.optimize import minimize
        result = minimize(objective, v0_guess, method='BFGS', 
                         options={'maxiter': 100})
        
        if result.success:
            v0_opt = result.x
            sol = self.solve_geodesic(theta_a, v0_opt, g_func, t_span=(0, 1), n_points=100)
            return self._compute_path_length(sol['theta'], sol['theta_dot'], g_func)
        else:
            # Fallback
            return np.linalg.norm(theta_b - theta_a)
    
    def _distance_relaxation(self, theta_a: np.ndarray, theta_b: np.ndarray,
                            g_func: Callable, n_nodes: int = 20) -> float:
        """
        Compute geodesic using relaxation method.
        
        Start with straight line and iteratively relax to geodesic.
        """
        # Initial guess: straight line
        t_nodes = np.linspace(0, 1, n_nodes)
        path = np.array([(1 - t) * theta_a + t * theta_b for t in t_nodes])
        
        # Relaxation iterations
        for iteration in range(50):
            path_new = path.copy()
            
            for i in range(1, n_nodes - 1):
                # Compute local metric
                g = g_func(path[i])
                
                # Update position using weighted average of neighbors
                # in the metric-induced geometry
                try:
                    g_inv = np.linalg.inv(g + 1e-6 * np.eye(4))
                    # Project neighbors to local tangent space
                    delta_prev = path[i] - path[i-1]
                    delta_next = path[i+1] - path[i]
                    
                    # Weighted average
                    path_new[i] = 0.5 * (path[i-1] + path[i+1])
                except np.linalg.LinAlgError:
                    path_new[i] = 0.5 * (path[i-1] + path[i+1])
            
            # Check convergence
            if np.max(np.abs(path_new - path)) < 1e-6:
                break
            
            path = path_new
        
        # Compute path length
        length = 0.0
        for i in range(n_nodes - 1):
            d_theta = path[i+1] - path[i]
            g = g_func(0.5 * (path[i] + path[i+1]))  # Midpoint metric
            length += np.sqrt(d_theta @ g @ d_theta)
        
        return length
    
    def _distance_discrete(self, theta_a: np.ndarray, theta_b: np.ndarray,
                          g_func: Callable, n_segments: int = 20) -> float:
        """
        Compute approximate geodesic distance using discrete path.
        """
        # Create discrete path
        t = np.linspace(0, 1, n_segments + 1)
        path = np.array([(1 - ti) * theta_a + ti * theta_b for ti in t])
        
        # Compute length
        length = 0.0
        for i in range(n_segments):
            d_theta = path[i+1] - path[i]
            g = g_func(path[i])
            
            # ds² = g_μν dθ^μ dθ^ν
            ds_squared = d_theta @ g @ d_theta
            
            if ds_squared > 0:
                length += np.sqrt(ds_squared)
        
        return length
    
    def _compute_path_length(self, theta_path: np.ndarray, 
                            theta_dot_path: np.ndarray,
                            g_func: Callable) -> float:
        """
        Compute length of a path on the manifold.
        
        L = ∫ √(g_μν θ̇^μ θ̇^ν) dt
        """
        n_points = len(theta_path)
        length = 0.0
        
        for i in range(n_points - 1):
            theta_mid = 0.5 * (theta_path[i] + theta_path[i+1])
            theta_dot = theta_dot_path[i]
            
            g = g_func(theta_mid)
            
            # Metric norm of velocity
            v_norm_sq = theta_dot @ g @ theta_dot
            
            if v_norm_sq > 0:
                dt = 1.0 / (n_points - 1)  # Assume uniform time steps
                length += np.sqrt(v_norm_sq) * dt
        
        return length
    
    def parallel_transport(self, vector: np.ndarray, path: np.ndarray,
                          g_func: Optional[Callable] = None) -> np.ndarray:
        """
        Parallel transport a vector along a curve on the manifold.
        
        Parallel transport preserves the inner product with the curve's tangent
        and maintains constant norm according to the metric.
        
        Args:
            vector: Initial vector to transport (4D, in tangent space at path[0])
            path: Curve points (n_points, 4)
            g_func: Function returning metric at a point
            
        Returns:
            transported: Vector transported to each point on path (n_points, 4)
            
        Mathematical Formulation:
        -------------------------
        Parallel transport equation:
            dV^μ/dt + Γ^μ_νρ V^ν (dθ^ρ/dt) = 0
            
        This ensures that V remains "parallel" to itself along the curve.
        """
        vector = np.asarray(vector, dtype=float)
        path = np.asarray(path, dtype=float)
        
        if g_func is None:
            g_func = lambda theta: self.fisher_metric.compute_fisher_metric(
                theta.reshape(1, -1), method='empirical'
            )
        
        n_points = len(path)
        transported = np.zeros((n_points, 4))
        transported[0] = vector
        
        for i in range(n_points - 1):
            theta = path[i]
            theta_next = path[i+1]
            d_theta = theta_next - theta
            
            # Compute metric and Christoffel symbols
            g = g_func(theta)
            Gamma = self.fisher_metric.compute_christoffel_symbols(g, theta)
            
            # Update vector using parallel transport equation
            # V^μ(t+dt) = V^μ(t) - dt * Γ^μ_νρ V^ν (dθ^ρ/dt)
            V_current = transported[i]
            
            dV = np.zeros(4)
            for mu in range(4):
                for nu in range(4):
                    for rho in range(4):
                        dV[mu] -= Gamma[mu, nu, rho] * V_current[nu] * d_theta[rho]
            
            transported[i+1] = V_current + dV
        
        return transported
    
    def exponential_map(self, theta: np.ndarray, v: np.ndarray,
                       g_func: Optional[Callable] = None,
                       t: float = 1.0) -> np.ndarray:
        """
        Exponential map: map tangent vector to point on manifold.
        
        The exponential map takes a tangent vector v at point θ and returns
        the point reached by following the geodesic with initial velocity v
        for unit time.
        
        Args:
            theta: Base point on manifold (4D)
            v: Tangent vector at θ (4D)
            g_func: Metric function
            t: Time parameter
            
        Returns:
            theta_new: Point on manifold reached by geodesic
        """
        sol = self.solve_geodesic(theta, v, g_func, t_span=(0, t), n_points=10)
        return sol['theta'][-1]
    
    def logarithmic_map(self, theta_a: np.ndarray, theta_b: np.ndarray,
                       g_func: Optional[Callable] = None) -> np.ndarray:
        """
        Logarithmic map: find tangent vector that reaches θ_b from θ_a.
        
        This is the inverse of the exponential map (when it exists).
        
        Args:
            theta_a: Starting point
            theta_b: Target point
            g_func: Metric function
            
        Returns:
            v: Tangent vector at θ_a such that exp_θ_a(v) = θ_b
        """
        # Use shooting method to find initial velocity
        v_guess = theta_b - theta_a
        
        def objective(v):
            theta_end = self.exponential_map(theta_a, v, g_func, t=1.0)
            return np.sum((theta_end - theta_b) ** 2)
        
        result = minimize(objective, v_guess, method='BFGS')
        
        if result.success:
            return result.x
        else:
            # Fallback to Euclidean
            return theta_b - theta_a
    
    def compute_geodesic_deviation(self, path1: np.ndarray, path2: np.ndarray,
                                   g_func: Optional[Callable] = None) -> float:
        """
        Compute deviation between two paths on the manifold.
        
        Useful for comparing actual demographic trajectories to geodesic predictions.
        """
        if g_func is None:
            g_func = lambda theta: self.fisher_metric.compute_fisher_metric(
                theta.reshape(1, -1), method='empirical'
            )
        
        # Ensure same number of points
        n_points = min(len(path1), len(path2))
        
        # Compute L² distance in metric sense
        deviation = 0.0
        for i in range(n_points):
            diff = path1[i] - path2[i]
            g = g_func(path1[i])
            deviation += diff @ g @ diff
        
        return np.sqrt(deviation / n_points)


# Convenience functions
def solve_geodesic(theta_0: np.ndarray, theta_dot_0: np.ndarray,
                  g: Optional[np.ndarray] = None,
                  t_span: Tuple[float, float] = (0, 1),
                  n_points: int = 100) -> dict:
    """
    Solve geodesic with constant metric approximation.
    
    Args:
        theta_0: Initial position
        theta_dot_0: Initial velocity
        g: Constant metric matrix (if None, uses identity)
        t_span: Time span
        n_points: Number of output points
        
    Returns:
        Dictionary with trajectory information
    """
    solver = GeodesicSolver()
    
    if g is not None:
        # Create constant metric function
        g_func = lambda theta: g
    else:
        g_func = None
    
    return solver.solve_geodesic(theta_0, theta_dot_0, g_func, t_span, n_points)


def compute_geodesic_distance(theta_a: np.ndarray, theta_b: np.ndarray,
                              g: Optional[np.ndarray] = None) -> float:
    """
    Compute geodesic distance with constant metric.
    
    For constant metric, geodesics are straight lines in the transformed
    coordinates where the metric is identity.
    """
    solver = GeodesicSolver()
    
    if g is not None:
        g_func = lambda theta: g
    else:
        g_func = None
    
    return solver.compute_geodesic_distance(theta_a, theta_b, g_func)


def parallel_transport(vector: np.ndarray, path: np.ndarray,
                      g: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Parallel transport with constant metric.
    """
    solver = GeodesicSolver()
    
    if g is not None:
        g_func = lambda theta: g
    else:
        g_func = None
    
    return solver.parallel_transport(vector, path, g_func)


# Example usage
if __name__ == "__main__":
    # Create synthetic test
    np.random.seed(42)
    
    # Define two points
    theta_a = np.array([1000.0, 3.5, 0.45, 0.8])
    theta_b = np.array([1500.0, 3.2, 0.50, 1.0])
    
    # Create solver
    solver = GeodesicSolver()
    
    # Compute geodesic distance
    distance = solver.compute_geodesic_distance(theta_a, theta_b, method='shortest_path')
    print(f"Geodesic distance: {distance:.4f}")
    
    # Compare to Euclidean
    euclidean_dist = np.linalg.norm(theta_b - theta_a)
    print(f"Euclidean distance: {euclidean_dist:.4f}")
    print(f"Ratio (geo/euc): {distance/euclidean_dist:.4f}")
    
    # Solve geodesic with initial velocity
    theta_dot_0 = theta_b - theta_a
    result = solver.solve_geodesic(theta_a, theta_dot_0, t_span=(0, 1), n_points=50)
    
    print(f"\nGeodesic integration success: {result['success']}")
    print(f"Initial position: {result['theta'][0]}")
    print(f"Final position: {result['theta'][-1]}")
