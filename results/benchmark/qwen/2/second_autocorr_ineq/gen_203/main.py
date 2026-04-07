# EVOLVE-BLOCK-START

import numpy as np
import warnings
from scipy.optimize import differential_evolution
from scipy import signal
import numba
from scipy.optimize import minimize
import optuna
from typing import List
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

@numba.jit(nopython=True)
def compute_autoconvolution_fast(f_vals):
    """Fast autoconvolution computation using Numba"""
    n = len(f_vals)
    g = np.zeros(2 * n - 1)
    
    # Manual convolution for speed
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]
    
    return g

@numba.jit(nopython=True)
def compute_norms_piecewise(g_vals):
    """Compute norms using piecewise linear integration matching evaluator's method"""
    n = len(g_vals)
    
    if n <= 1:
        return 0.0, 0.0, 0.0
    
    # Compute L2 norm squared using trapezoidal-like integration
    # Formula: (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
    norm_2_sq = 0.0
    dx = 0.5 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.5
    
    for i in range(n - 1):
        y1 = g_vals[i]
        y2 = g_vals[i + 1]
        norm_2_sq += (dx / 3.0) * (y1 * y1 + y1 * y2 + y2 * y2)
    
    # Compute L1 norm (sum of absolute values)
    norm_1 = 0.0
    for i in range(n):
        norm_1 += abs(g_vals[i])
    
    # Compute L-infinity norm (maximum absolute value)
    norm_inf = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > norm_inf:
            norm_inf = abs_val
    
    return norm_2_sq, norm_1, norm_inf

def compute_autoconvolution_norms(f: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation using efficient piecewise integration.
    Returns (||g||₂², ||g||₁, ||g||∞)
    """
    # Convert to numpy array
    f_arr = np.array(f, dtype=np.float64)
    
    # Compute autoconvolution
    g = compute_autoconvolution_fast(f_arr)
    
    # Compute norms using piecewise integration
    norm_2_sq, norm_1, norm_inf = compute_norms_piecewise(g)
    
    return norm_2_sq, norm_1, norm_inf

def compute_c2(f: List[float]) -> float:
    """Compute C2 value for given function"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f)
    
    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    
    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def generate_spectral_basis_functions(n_points: int, n_components: int = 12) -> np.ndarray:
    """
    Generate a set of basis functions in frequency domain representing sinusoidal components
    that can be combined to form step functions in time domain
    """
    # Create frequency grid for basis functions
    freqs = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Generate basis functions: sine/cosine combinations with different frequencies
    basis_functions = []
    
    # Base frequencies (logarithmic spacing in frequency)
    base_freqs = np.logspace(np.log10(0.5), np.log10(n_points//4), n_components, endpoint=True)
    
    for i, freq in enumerate(base_freqs):
        # Generate different types of basis functions with varying phase
        # Cosine basis
        basis_cos = np.cos(freq * freqs)
        # Sine basis  
        basis_sin = np.sin(freq * freqs)
        
        # Scale by inverse frequency to control energy distribution
        scale_factor = 1.0 / (1.0 + freq**0.5)
        
        basis_functions.append(basis_cos * scale_factor)
        basis_functions.append(basis_sin * scale_factor)
    
    return np.array(basis_functions)

def spectral_function_constructor(coefficients: np.ndarray, n_points: int = 1000) -> np.ndarray:
    """
    Construct a step function from spectral coefficients using basis functions
    """
    # Generate basis functions
    basis = generate_spectral_basis_functions(n_points)
    
    # Ensure we have enough coefficients for the basis
    n_basis = basis.shape[0]
    if len(coefficients) < n_basis:
        coeffs_padded = np.pad(coefficients, (0, n_basis - len(coefficients)), 'constant')
    else:
        coeffs_padded = coefficients[:n_basis]
    
    # Linear combination of basis functions
    f_vals = np.zeros(n_points)
    for i in range(n_basis):
        # Apply coefficient and add to function
        f_vals += coeffs_padded[i] * basis[i]
    
    # Ensure non-negativity
    f_vals = np.maximum(f_vals, 0)
    
    # Normalize to reasonable scale
    if np.max(f_vals) > 0:
        f_vals = f_vals / np.max(f_vals) * 2.0
    
    return f_vals

def objective_function(params: np.ndarray) -> float:
    """
    Objective function to maximize C2 score
    """
    try:
        # Convert to function using spectral basis
        n_points = 1000
        f_vals = spectral_function_constructor(params, n_points)
        
        # Compute C2 score
        c2_score = compute_c2(f_vals.tolist())
        
        # Return negative because we're minimizing in scipy
        return -c2_score if c2_score > 0 else 1e10
    except:
        return 1e10

def bayesian_optimization_approach() -> List[float]:
    """
    Use Bayesian optimization for spectral parameter optimization
    """
    # Parameters to optimize (coefficients of basis functions)
    n_params = 20
    
    # Define optimization bounds
    bounds = [(-2.0, 2.0) for _ in range(n_params)]
    
    # Use Optuna for Bayesian optimization
    def objective(trial):
        params = []
        for i in range(n_params):
            param = trial.suggest_uniform(f'param_{i}', bounds[i][0], bounds[i][1])
            params.append(param)
        
        # Compute C2 score
        try:
            f_vals = spectral_function_constructor(np.array(params), 1000)
            c2_score = compute_c2(f_vals.tolist())
            return c2_score if c2_score > 0 else 0.0
        except:
            return 0.0
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, timeout=80)
    
    if study.best_trial:
        best_params = [study.best_trial.params[f'param_{i}'] for i in range(n_params)]
        
        # Generate final function
        f_vals = spectral_function_constructor(np.array(best_params), 1000)
        return f_vals.tolist()
    
    # Fallback
    return [1.0] * 1000

def construct_function() -> List[float]:
    """
    Main function to construct step function with high C2 value
    Using spectral domain optimization approach
    """
    start_time = time.time()
    
    # Primary approach: Bayesian optimization
    try:
        best_function = bayesian_optimization_approach()
        
        # Validate and refine if necessary
        if best_function and len(best_function) > 0:
            # Run a quick local refinement if time permits
            if time.time() - start_time < 80:
                # Simple refinement: small perturbations to coefficients
                current_params = np.array([1.0] * 20)  # placeholder
                try:
                    # Quick local search around best parameters
                    def refine_objective(perturbed_params):
                        try:
                            f_vals = spectral_function_constructor(perturbed_params, 1000)
                            c2_score = compute_c2(f_vals.tolist())
                            return -c2_score if c2_score > 0 else 1e10
                        except:
                            return 1e10
                    
                    # Simple gradient-free search
                    best_c2 = compute_c2(best_function)
                    for _ in range(10):
                        # Perturb slightly
                        perturbed = current_params + np.random.normal(0, 0.1, len(current_params))
                        # Evaluate
                        new_c2 = -refine_objective(perturbed)
                        if new_c2 > best_c2:
                            best_c2 = new_c2
                            current_params = perturbed.copy()
                    
                    # Recompute with best parameters found
                    f_vals = spectral_function_constructor(current_params, 1000)
                    if compute_c2(f_vals.tolist()) > best_c2:
                        best_function = f_vals.tolist()
                        
                except:
                    pass
                    
        return best_function
    except Exception as e:
        # Fallback to simpler approach if Bayesian optimization fails
        n_points = 500
        # Create a simple structured function
        f_vals = np.zeros(n_points)
        # Add some peaks
        for i in range(5):
            center = n_points // 4 + i * n_points // 8
            width = n_points // 50
            height = 1.0 + i * 0.2
            x = np.arange(n_points)
            gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
            f_vals += gaussian
        
        f_vals = np.maximum(f_vals, 0)
        if np.max(f_vals) > 0:
            f_vals = f_vals / np.max(f_vals) * 2.0
        
        return f_vals.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")