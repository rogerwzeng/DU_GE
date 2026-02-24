"""
Shock Classification Module
===========================

Classify urban shocks as adiabatic (resilient) vs non-adiabatic (transformational).

Mathematical formalism from Dissipative Urbanism thesis:

1. Quench Rate: Q = |dλ/dt| / τ_system
   where λ is control parameter, τ_system ≈ 2-3 years (demographic relaxation)

2. Shock Classification:
   - Adiabatic (Resilient): System tracks equilibrium manifold
   - Non-Adiabatic (Transformational): System departs manifold, bifurcates

3. Detection Criteria:
   - Trend deviation: Permanent shift from pre-shock trend
   - Variance inflation: Exceeds 2σ of historical variance
   - Recovery rate: λ_recovery < threshold indicates non-adiabatic
   - Manifold tearing: Discontinuity in trajectory (Fisher metric)
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass
from enum import Enum
import warnings

from critical_slowing_down import (
    compute_ews_metrics,
    estimate_recovery_rate,
    test_trend_significance
)


class ShockType(Enum):
    """Classification of shock response types."""
    ADIABATIC = "adiabatic"           # Resilient - returns to equilibrium
    NON_ADIABATIC = "non_adiabatic"   # Transformational - new attractor
    UNCERTAIN = "uncertain"           # Insufficient data to classify
    NO_SHOCK = "no_shock"             # No significant shock detected


@dataclass
class ShockClassification:
    """Container for shock classification results."""
    msa_code: str
    msa_name: str
    shock_date: int  # Year of shock
    shock_type: ShockType
    confidence: float  # 0-1 classification confidence
    
    # Classification metrics
    quench_rate: float
    trend_deviation: float
    variance_ratio: float
    recovery_rate: float
    
    # Bifurcation indicators
    bifurcation_detected: bool
    manifold_tearing: bool
    permanent_shift: bool
    
    # Recovery metrics
    recovery_time: Optional[int]  # Years to return to trend (None if never)
    new_equilibrium: Optional[float]  # New equilibrium level (if bifurcation)
    
    # Comparison to pre-shock
    pre_shock_trend: float
    post_shock_trend: float
    trend_change: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'msa_code': self.msa_code,
            'msa_name': self.msa_name,
            'shock_date': self.shock_date,
            'shock_type': self.shock_type.value,
            'confidence': self.confidence,
            'quench_rate': self.quench_rate,
            'trend_deviation': self.trend_deviation,
            'variance_ratio': self.variance_ratio,
            'recovery_rate': self.recovery_rate,
            'bifurcation_detected': self.bifurcation_detected,
            'manifold_tearing': self.manifold_tearing,
            'permanent_shift': self.permanent_shift,
            'recovery_time': self.recovery_time,
            'new_equilibrium': self.new_equilibrium,
            'pre_shock_trend': self.pre_shock_trend,
            'post_shock_trend': self.post_shock_trend,
            'trend_change': self.trend_change
        }


def compute_quench_rate(
    control_parameter: Union[pd.Series, np.ndarray],
    tau_system: float = 2.5,
    shock_idx: Optional[int] = None
) -> float:
    """
    Compute quench rate Q = |dλ/dt| / τ_system.
    
    The quench rate measures how rapidly the control parameter changes
    relative to the system's natural relaxation time.
    
    Parameters:
    -----------
    control_parameter : array-like
        Time series of control parameter (e.g., population growth rate)
    tau_system : float, default 2.5
        System relaxation time in years (demographic ~ 2-3 years)
    shock_idx : int, optional
        Index of shock event. If None, uses maximum rate of change.
        
    Returns:
    --------
    float
        Quench rate Q (dimensionless)
        
    Interpretation:
    - Q << 1: Adiabatic regime (slow change, system stays near equilibrium)
    - Q ~ 1: Critical regime (transitional)
    - Q >> 1: Non-adiabatic (rapid change, system departs from equilibrium)
    """
    if isinstance(control_parameter, pd.Series):
        cp = control_parameter.dropna().values
    else:
        cp = np.array(control_parameter)
        cp = cp[~np.isnan(cp)]
    
    if len(cp) < 2:
        return np.nan
    
    # Compute rate of change
    d_lambda = np.diff(cp)
    
    if shock_idx is not None and 0 < shock_idx < len(d_lambda):
        # Use rate at shock
        max_rate = np.abs(d_lambda[shock_idx])
    else:
        # Use maximum rate of change
        max_rate = np.max(np.abs(d_lambda))
    
    # Quench rate = |dλ/dt| / τ_system
    Q = max_rate / tau_system
    
    return Q


def detect_bifurcation(
    time_series: Union[pd.Series, np.ndarray],
    pre_window: int = 3,
    post_window: int = 3,
    shock_idx: Optional[int] = None,
    significance: float = 0.05
) -> Dict[str, Union[bool, float]]:
    """
    Detect bifurcation using pre/post comparison.
    
    A bifurcation is detected when post-shock dynamics differ
    significantly from pre-shock dynamics.
    
    Parameters:
    -----------
    time_series : array-like
        Time series data
    pre_window : int, default 3
        Number of years before shock for baseline
    post_window : int, default 3
        Number of years after shock for comparison
    shock_idx : int, optional
        Index of shock event. If None, uses middle of series.
    significance : float, default 0.05
        Significance level for statistical tests
        
    Returns:
    --------
    dict
        Dictionary with bifurcation indicators:
        - 'bifurcation_detected': Boolean
        - 'p_value': Statistical significance
        - 'mean_shift': Shift in mean level
        - 'variance_shift': Change in variance
        - 'regime_change_score': Composite score
    """
    if isinstance(time_series, pd.Series):
        ts = time_series.dropna().values
    else:
        ts = np.array(time_series)
    
    n = len(ts)
    
    if shock_idx is None:
        shock_idx = n // 2
    
    # Ensure valid windows
    pre_start = max(0, shock_idx - pre_window)
    pre_end = shock_idx
    post_start = shock_idx + 1
    post_end = min(n, shock_idx + 1 + post_window)
    
    if pre_end - pre_start < 2 or post_end - post_start < 2:
        return {
            'bifurcation_detected': False,
            'p_value': 1.0,
            'mean_shift': 0.0,
            'variance_shift': 1.0,
            'regime_change_score': 0.0
        }
    
    pre_data = ts[pre_start:pre_end]
    post_data = ts[post_start:post_end]
    
    # Statistical tests
    # 1. Mean shift (t-test)
    t_stat, p_value_mean = stats.ttest_ind(pre_data, post_data)
    
    # 2. Variance shift (Levene's test)
    _, p_value_var = stats.levene(pre_data, post_data)
    
    # 3. Distribution shift (Kolmogorov-Smirnov test)
    _, p_value_ks = stats.ks_2samp(pre_data, post_data)
    
    # Compute effect sizes
    mean_shift = np.mean(post_data) - np.mean(pre_data)
    pooled_std = np.sqrt((np.std(pre_data)**2 + np.std(post_data)**2) / 2)
    cohens_d = mean_shift / pooled_std if pooled_std > 0 else 0
    
    variance_shift = np.var(post_data) / np.var(pre_data) if np.var(pre_data) > 0 else 1.0
    
    # Regime change score (composite)
    # High score = significant change in both mean and variance
    regime_change_score = (
        (1 - p_value_mean) * abs(cohens_d) +
        (1 - p_value_ks) +
        abs(np.log(variance_shift))
    ) / 3
    
    # Bifurcation detection criteria
    bifurcation_detected = (
        p_value_mean < significance and
        abs(cohens_d) > 0.5 and
        regime_change_score > 0.5
    )
    
    return {
        'bifurcation_detected': bifurcation_detected,
        'p_value': p_value_mean,
        'mean_shift': mean_shift,
        'variance_shift': variance_shift,
        'cohens_d': cohens_d,
        'regime_change_score': regime_change_score
    }


def detect_manifold_tearing(
    time_series: Union[pd.Series, np.ndarray],
    shock_idx: Optional[int] = None,
    threshold: float = 2.0
) -> Dict[str, Union[bool, float]]:
    """
    Detect manifold tearing (discontinuity in trajectory).
    
    Manifold tearing occurs when a shock causes a discontinuous jump
    in the system's state space trajectory, violating the smooth
    manifold assumption of adiabatic evolution.
    
    Parameters:
    -----------
    time_series : array-like
        Time series data
    shock_idx : int, optional
        Index of shock event
    threshold : float, default 2.0
        Threshold for discontinuity detection (in units of std dev)
        
    Returns:
    --------
    dict
        Dictionary with manifold tearing indicators:
        - 'manifold_tearing': Boolean
        - 'discontinuity_score': Magnitude of discontinuity
        - 'jump_magnitude': Absolute jump size
        - 'normalized_jump': Jump normalized by local variance
    """
    if isinstance(time_series, pd.Series):
        ts = time_series.dropna().values
    else:
        ts = np.array(time_series)
    
    n = len(ts)
    
    if shock_idx is None:
        shock_idx = n // 2
    
    if shock_idx <= 0 or shock_idx >= n - 1:
        return {
            'manifold_tearing': False,
            'discontinuity_score': 0.0,
            'jump_magnitude': 0.0,
            'normalized_jump': 0.0
        }
    
    # Compute local trend before shock
    pre_window = min(3, shock_idx)
    if pre_window >= 2:
        x_pre = np.arange(pre_window)
        y_pre = ts[shock_idx - pre_window:shock_idx]
        slope, intercept = np.polyfit(x_pre, y_pre, 1)
        expected_value = slope * pre_window + intercept
    else:
        expected_value = ts[shock_idx - 1]
    
    # Actual value at shock
    actual_value = ts[shock_idx]
    
    # Jump magnitude
    jump_magnitude = actual_value - expected_value
    
    # Local variance for normalization
    local_std = np.std(ts[max(0, shock_idx-3):min(n, shock_idx+4)])
    normalized_jump = abs(jump_magnitude) / local_std if local_std > 0 else 0
    
    # Discontinuity score
    discontinuity_score = normalized_jump
    
    # Manifold tearing detection
    manifold_tearing = normalized_jump > threshold
    
    return {
        'manifold_tearing': manifold_tearing,
        'discontinuity_score': discontinuity_score,
        'jump_magnitude': jump_magnitude,
        'normalized_jump': normalized_jump
    }


def estimate_recovery_time(
    time_series: Union[pd.Series, np.ndarray],
    shock_idx: int,
    max_years: int = 10,
    tolerance: float = 0.05
) -> Optional[int]:
    """
    Estimate time to return to pre-shock trend.
    
    Parameters:
    -----------
    time_series : array-like
        Time series data
    shock_idx : int
        Index of shock event
    max_years : int, default 10
        Maximum years to look for recovery
    tolerance : float, default 0.05
        Fractional tolerance for "recovery" (5% of pre-shock level)
        
    Returns:
    --------
    int or None
        Years to recovery, or None if no recovery within max_years
    """
    if isinstance(time_series, pd.Series):
        ts = time_series.dropna().values
    else:
        ts = np.array(time_series)
    
    n = len(ts)
    
    if shock_idx <= 0 or shock_idx >= n:
        return None
    
    # Compute pre-shock trend
    pre_window = min(3, shock_idx)
    if pre_window < 2:
        return None
    
    x_pre = np.arange(pre_window)
    y_pre = ts[shock_idx - pre_window:shock_idx]
    slope, intercept = np.polyfit(x_pre, y_pre, 1)
    
    # Extrapolate trend
    trend_values = slope * np.arange(max_years + 1) + ts[shock_idx - 1] + slope
    
    # Find recovery point
    for t in range(1, min(max_years + 1, n - shock_idx)):
        actual_idx = shock_idx + t
        if actual_idx >= n:
            break
        
        deviation = abs(ts[actual_idx] - trend_values[t])
        tolerance_val = tolerance * abs(trend_values[t])
        
        if deviation < tolerance_val:
            return t
    
    return None  # No recovery within max_years


def classify_shock_response(
    msa_data: pd.DataFrame,
    shock_date: int,
    recovery_window: int = 5,
    population_col: str = 'population',
    year_col: str = 'year',
    msa_code: Optional[str] = None,
    msa_name: Optional[str] = None
) -> ShockClassification:
    """
    Classify shock response as adiabatic or non-adiabatic.
    
    Parameters:
    -----------
    msa_data : pd.DataFrame
        DataFrame with time series data for an MSA
    shock_date : int
        Year of shock event
    recovery_window : int, default 5
        Years to examine for recovery
    population_col : str, default 'population'
        Column name for population data
    year_col : str, default 'year'
        Column name for year
    msa_code : str, optional
        MSA code
    msa_name : str, optional
        MSA name
        
    Returns:
    --------
    ShockClassification
        Classification result with all metrics
    """
    # Sort by year
    data = msa_data.sort_values(year_col).copy()
    
    # Extract series
    years = data[year_col].values
    population = data[population_col].values
    
    # Find shock index
    shock_mask = years == shock_date
    if not shock_mask.any():
        # Find closest year
        shock_idx = np.argmin(np.abs(years - shock_date))
        shock_year = years[shock_idx]
    else:
        shock_idx = np.where(shock_mask)[0][0]
        shock_year = shock_date
    
    # Get MSA info
    if msa_code is None:
        msa_code = str(data.iloc[0].get('msa_code', 'Unknown'))
    if msa_name is None:
        msa_name = str(data.iloc[0].get('msa_name', 'Unknown'))
    
    # Compute population growth rate (control parameter)
    growth_rate = np.diff(population) / population[:-1]
    
    # 1. Quench rate
    quench_rate = compute_quench_rate(growth_rate, shock_idx=shock_idx)
    
    # 2. Bifurcation detection
    bifurcation_result = detect_bifurcation(
        population, 
        pre_window=3, 
        post_window=recovery_window,
        shock_idx=shock_idx
    )
    
    # 3. Manifold tearing
    tearing_result = detect_manifold_tearing(
        population,
        shock_idx=shock_idx
    )
    
    # 4. Recovery time
    recovery_time = estimate_recovery_time(
        population,
        shock_idx=shock_idx,
        max_years=recovery_window
    )
    
    # 5. Trend analysis
    pre_data = data[data[year_col] < shock_year]
    post_data = data[data[year_col] > shock_year]
    
    if len(pre_data) >= 2 and len(post_data) >= 2:
        # Fit trends
        pre_slope, _ = np.polyfit(pre_data[year_col], pre_data[population_col], 1)
        post_slope, post_intercept = np.polyfit(post_data[year_col], post_data[population_col], 1)
        
        # Pre-shock trend value at shock point
        pre_shock_trend = pre_slope * shock_year
        # Post-shock trend value at shock point  
        post_shock_trend = post_slope * shock_year + post_intercept
        
        trend_change = post_slope - pre_slope
        trend_deviation = post_shock_trend - pre_shock_trend
    else:
        pre_shock_trend = np.nan
        post_shock_trend = np.nan
        trend_change = np.nan
        trend_deviation = np.nan
    
    # 6. Compute EWS metrics
    ews = compute_ews_metrics(population, window=min(5, len(population)//2))
    
    # Get recovery rate near shock
    if shock_idx < len(ews['recovery_rate']):
        recovery_rate = ews['recovery_rate'].iloc[shock_idx]
        if pd.isna(recovery_rate) and shock_idx > 0:
            # Try nearby values
            nearby = ews['recovery_rate'].iloc[max(0, shock_idx-2):min(len(ews), shock_idx+3)]
            recovery_rate = nearby.mean()
    else:
        recovery_rate = np.nan
    
    # Variance ratio (post-shock / pre-shock)
    pre_var = np.var(population[:shock_idx]) if shock_idx > 0 else np.nan
    post_var = np.var(population[shock_idx:]) if shock_idx < len(population) else np.nan
    variance_ratio = post_var / pre_var if pre_var > 0 else 1.0
    
    # Classification logic
    # Non-adiabatic indicators:
    # - High quench rate (Q > 0.5)
    # - Bifurcation detected
    # - Manifold tearing
    # - No recovery within window
    # - Permanent trend shift
    
    non_adiabatic_score = 0.0
    adiabatic_score = 0.0
    
    # Quench rate criterion
    if quench_rate > 0.5:
        non_adiabatic_score += 0.2
    elif quench_rate < 0.2:
        adiabatic_score += 0.2
    
    # Bifurcation criterion
    if bifurcation_result['bifurcation_detected']:
        non_adiabatic_score += 0.25
    else:
        adiabatic_score += 0.25
    
    # Manifold tearing criterion
    if tearing_result['manifold_tearing']:
        non_adiabatic_score += 0.2
    else:
        adiabatic_score += 0.2
    
    # Recovery criterion
    if recovery_time is None:
        non_adiabatic_score += 0.2  # No recovery = non-adiabatic
    elif recovery_time <= 2:
        adiabatic_score += 0.2  # Quick recovery = adiabatic
    
    # Trend shift criterion
    if not np.isnan(trend_change):
        if abs(trend_change) > 0.05 * abs(pre_shock_trend) if pre_shock_trend != 0 else abs(trend_change) > 1000:
            non_adiabatic_score += 0.15
        else:
            adiabatic_score += 0.15
    
    # Classification
    total_score = non_adiabatic_score + adiabatic_score
    if total_score > 0:
        non_adiabatic_prob = non_adiabatic_score / total_score
    else:
        non_adiabatic_prob = 0.5
    
    if non_adiabatic_prob > 0.6:
        shock_type = ShockType.NON_ADIABATIC
        confidence = non_adiabatic_prob
    elif non_adiabatic_prob < 0.4:
        shock_type = ShockType.ADIABATIC
        confidence = 1 - non_adiabatic_prob
    else:
        shock_type = ShockType.UNCERTAIN
        confidence = 0.5
    
    # Uncertain if insufficient data
    if len(data) < 5 or shock_idx < 2 or shock_idx >= len(data) - 2:
        shock_type = ShockType.UNCERTAIN
        confidence = 0.3
    
    return ShockClassification(
        msa_code=msa_code,
        msa_name=msa_name,
        shock_date=shock_year,
        shock_type=shock_type,
        confidence=confidence,
        quench_rate=quench_rate,
        trend_deviation=trend_deviation,
        variance_ratio=variance_ratio,
        recovery_rate=recovery_rate if not pd.isna(recovery_rate) else np.nan,
        bifurcation_detected=bifurcation_result['bifurcation_detected'],
        manifold_tearing=tearing_result['manifold_tearing'],
        permanent_shift=recovery_time is None and bifurcation_result['bifurcation_detected'],
        recovery_time=recovery_time,
        new_equilibrium=np.mean(population[shock_idx:]) if bifurcation_result['bifurcation_detected'] else None,
        pre_shock_trend=pre_shock_trend if not np.isnan(pre_shock_trend) else np.nan,
        post_shock_trend=post_shock_trend if not np.isnan(post_shock_trend) else np.nan,
        trend_change=trend_change if not np.isnan(trend_change) else np.nan
    )


def classify_multiple_msas(
    data: pd.DataFrame,
    shock_date: int,
    recovery_window: int = 5,
    msa_code_col: str = 'msa_code',
    msa_name_col: str = 'msa_name'
) -> pd.DataFrame:
    """
    Classify shock response for multiple MSAs.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with data for all MSAs
    shock_date : int
        Year of shock
    recovery_window : int, default 5
        Recovery window years
    msa_code_col : str, default 'msa_code'
        Column name for MSA code
    msa_name_col : str, default 'msa_name'
        Column name for MSA name
        
    Returns:
    --------
    pd.DataFrame
        Classification results for all MSAs
    """
    results = []
    
    for msa_code in data[msa_code_col].unique():
        msa_data = data[data[msa_code_col] == msa_code]
        msa_name = msa_data[msa_name_col].iloc[0] if msa_name_col in msa_data.columns else 'Unknown'
        
        try:
            classification = classify_shock_response(
                msa_data,
                shock_date=shock_date,
                recovery_window=recovery_window,
                msa_code=str(msa_code),
                msa_name=str(msa_name)
            )
            results.append(classification.to_dict())
        except Exception as e:
            warnings.warn(f"Error classifying MSA {msa_code}: {e}")
            continue
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    print("Shock Classifier Module - Example Usage")
    print("=" * 50)
    
    # Generate example data: adiabatic (resilient) response
    np.random.seed(42)
    years = np.arange(1995, 2010)
    trend = 1000000 + 5000 * (years - 1995)
    
    # Add shock in 2001 with quick recovery (adiabatic)
    shock_effect = np.zeros_like(years, dtype=float)
    shock_effect[years == 2001] = -20000
    shock_effect[years == 2002] = -10000
    
    population = trend + shock_effect + np.random.normal(0, 5000, len(years))
    
    df = pd.DataFrame({
        'year': years,
        'population': population,
        'msa_code': 'TEST001',
        'msa_name': 'Test MSA'
    })
    
    classification = classify_shock_response(df, shock_date=2001)
    
    print(f"\nMSA: {classification.msa_name}")
    print(f"Shock Type: {classification.shock_type.value}")
    print(f"Confidence: {classification.confidence:.2f}")
    print(f"Quench Rate: {classification.quench_rate:.3f}")
    print(f"Bifurcation Detected: {classification.bifurcation_detected}")
    print(f"Recovery Time: {classification.recovery_time} years")
