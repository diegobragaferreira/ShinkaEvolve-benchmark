# EVOLVE-BLOCK-START
import numpy as np
import time
from numba import jit
import optuna
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

# Global constants
N_BINS = 1000
DOMAIN = [-0.25, 0.25]
STEP_WIDTH = (DOMAIN[1] - DOMAIN[0]) / N_BINS

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using fast Numba implementation"""
    n = len(f_vals)
    # Convolution result has length 2*n-1
    g_len = 2 * n - 1
    g = np.zeros(g_len)

    # Compute convolution manually for efficiency
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_len:
                g[idx] += f_vals[i] * f_vals[j]

    return g

@jit(nopython=True)
def compute_c2_numba(g_vals):
    """Compute C2 value using fast Numba implementation"""
    if len(g_vals) == 0:
        return 0.0

    # Compute norms
    g_l2_sq = 0.0
    g_l1 = 0.0
    g_max = 0.0

    # For L2 norm squared using trapezoidal integration
    if len(g_vals) >= 2:
        # Trapezoidal rule for L2 norm squared: int g^2 dt ≈ h * (g[0]^2 + 2*sum(g[i]^2) + g[n-1]^2)/2
        g_l2_sq = g_vals[0]*g_vals[0] + g_vals[-1]*g_vals[-1]
        for i in range(1, len(g_vals)-1):
            g_l2_sq += 2 * g_vals[i] * g_vals[i]
        # Step width calculation for convolution
        h = 0.5 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.001
        g_l2_sq *= h / 2.0
    elif len(g_vals) == 1:
        g_l2_sq = g_vals[0] * g_vals[0]

    # For L1 norm (sum of absolute values)
    for i in range(len(g_vals)):
        g_l1 += abs(g_vals[i])

    # For infinity norm (max absolute value)
    for i in range(len(g_vals)):
        if abs(g_vals[i]) > g_max:
            g_max = abs(g_vals[i])

    # Compute C2
    if g_l1 > 1e-15 and g_max > 1e-15:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

def compute_c2_for_params(params):
    """Wrapper function for optuna optimization"""
    try:
        # Ensure non-negative values
        f_vals = np.clip(params, 0, None)

        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals)

        # Compute C2
        c2 = compute_c2_numba(g_vals)

        return c2
    except Exception:
        return 0.0

def convolution_aware_local_search(initial_params, max_iter=50):
    """Local search method that exploits the structure of convolution"""
    def objective(x):
        return -compute_c2_for_params(x)

    # Use L-BFGS-B for local refinement, which works well for smooth functions
    try:
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=[(0, 10) for _ in range(len(initial_params))],
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )
        return result.x
    except:
        return initial_params

def adaptive_optimization(trial, n_steps):
    """Adaptive optimization using optuna with dynamic exploration"""
    # Dynamic parameter sampling based on problem characteristics
    scales = []
    for i in range(n_steps):
        # Sample from different distributions based on position to capture structure
        if i % 3 == 0:
            # High peaks
            scale = trial.suggest_float(f'scale_{i}', 0.1, 3.0)
        elif i % 3 == 1:
            # Low valleys
            scale = trial.suggest_float(f'scale_{i}', 0.01, 0.5)
        else:
            # Medium levels
            scale = trial.suggest_float(f'scale_{i}', 0.1, 1.5)
        scales.append(scale)

    # Apply some structured patterns
    params = []
    for i, scale in enumerate(scales):
        # Create a pattern that favors certain structures known to work well
        pattern = 0.5 + 0.3 * np.sin(i * 0.5)
        if i % 4 == 0:
            pattern = 1.0
        elif i % 4 == 2:
            pattern = 0.2
        else:
            pattern = 0.5 + 0.3 * np.sin(i * 0.3)

        # Combine with scale and add some noise
        param = max(0, pattern * scale + np.random.normal(0, 0.1))
        params.append(param)

    return params

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using adaptive optimization."""
    start_time = time.time()

    # Set up optuna study for adaptive optimization
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )

    best_c2 = -np.inf
    best_params = None

    # Try different configurations to find best
    configurations = [
        (200, 50),
        (400, 50),
        (600, 50),
        (800, 50),
        (1000, 50)
    ]

    # Add some randomness to better explore
    for _ in range(10):
        dim = np.random.randint(200, 1000)
        iterations = np.random.randint(30, 60)
        configurations.append((dim, iterations))

    # Run optimization for each configuration
    for n_steps, n_trials in configurations:
        if time.time() - start_time > 85:
            break

        try:
            # Create temporary study for this configuration
            temp_study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=np.random.randint(1000)),
                pruner=optuna.pruners.MedianPruner()
            )

            # Run trials
            for trial_num in range(n_trials):
                if time.time() - start_time > 85:
                    break

                trial = temp_study.ask()

                # Get parameters for this trial
                params = adaptive_optimization(trial, n_steps)

                # Evaluate
                c2 = compute_c2_for_params(params)

                # Report result
                temp_study.tell(trial, c2)

                # Check if this is the best so far
                if c2 > best_c2:
                    best_c2 = c2
                    best_params = params.copy()

        except Exception as e:
            continue

    # Perform local refinement on the best found solution
    if best_params is not None and len(best_params) > 0:
        try:
            refined_params = convolution_aware_local_search(best_params, 100)
            refined_c2 = compute_c2_for_params(refined_params)

            if refined_c2 > best_c2:
                best_c2 = refined_c2
                best_params = refined_params
        except:
            pass

    # If still no valid parameters, fallback to structured initialization
    if best_params is None or len(best_params) == 0:
        # Create a pattern that's known to work reasonably well
        n_steps = 500
        best_params = []
        for i in range(n_steps):
            if i % 5 == 0:
                best_params.append(1.0)
            elif i % 5 == 2:
                best_params.append(0.3)
            else:
                best_params.append(0.7)

        # Add some noise for diversity
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, len(best_params))
        best_params = np.array(best_params) + noise
        best_params = np.maximum(best_params, 0).tolist()

    return best_params

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")