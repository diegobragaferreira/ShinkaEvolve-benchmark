# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import convolve
import numba
from typing import List

@numba.jit(nopython=True)
def compute_autoconvolution_norms(f_vals: np.ndarray, n_points: int = 10000) -> tuple:
    """
    Compute the norms for autoconvolution g = f*f using piecewise linear integration
    """
    # Create step function on [-1/4, 1/4] with given values
    step_width = 0.5 / len(f_vals)
    x = np.linspace(-0.25, 0.25, len(f_vals))

    # Create piecewise constant function
    f = np.zeros(n_points)
    x_grid = np.linspace(-0.25, 0.25, n_points)

    # Interpolate step function onto grid
    for i in range(len(f_vals)):
        start_idx = max(0, int((x[i] + 0.25) / step_width * n_points / 0.5))
        end_idx = min(n_points, int((x[i+1] if i+1 < len(x) else 0.25 + step_width) + 0.25) / step_width * n_points / 0.5)
        if i == len(f_vals) - 1:
            end_idx = n_points
        f[start_idx:end_idx] = f_vals[i]

    # Compute autoconvolution g = f * f (discrete convolution)
    g = convolve(f, f[::-1], mode='full')
    g = g[len(g)//2:]  # Take positive half

    # Truncate to match x_grid size
    g = g[:n_points]

    # Compute norms using trapezoidal-like piecewise integration
    # For ||g||_2^2: integrate g^2
    g_squared = g * g
    norm_g2_sq = np.sum((g_squared[:-1] + g_squared[1:]) * (x_grid[1] - x_grid[0]) / 2)

    # For ||g||_1
    norm_g1 = np.sum(np.abs(g)) * (x_grid[1] - x_grid[0])

    # For ||g||_inf
    norm_ginf = np.max(np.abs(g))

    return norm_g2_sq, norm_g1, norm_ginf

@numba.jit(nopython=True)
def evaluate_c2(f_vals: np.ndarray) -> float:
    """Evaluate C2 = ||g||_2^2 / (||g||_1 * ||g||_inf)"""
    norm_g2_sq, norm_g1, norm_ginf = compute_autoconvolution_norms(f_vals)

    # Handle numerical issues
    if norm_g1 < 1e-12 or norm_ginf < 1e-12:
        return 0.0

    return norm_g2_sq / (norm_g1 * norm_ginf)

def create_multiple_initializations(n_steps: int) -> List[List[float]]:
    """
    Create multiple diverse initializations to enhance exploration
    """
    initializations = []

    # 1. Gaussian shaped peaks
    x = np.linspace(0, 1, n_steps)
    gaussian1 = np.exp(-((x - 0.3)**2) / 0.05) * 0.8
    gaussian2 = np.exp(-((x - 0.7)**2) / 0.05) * 0.8
    initializations.append((gaussian1 + gaussian2).tolist())

    # 2. Uniform distribution
    initializations.append([1.0] * n_steps)

    # 3. Alternating pattern
    alt_pattern = []
    for i in range(n_steps):
        alt_pattern.append(1.0 if i % 2 == 0 else 0.1)
    initializations.append(alt_pattern)

    # 4. Center-heavy pattern
    center_pattern = []
    for i in range(n_steps):
        pos = i / (n_steps - 1) if n_steps > 1 else 0.5
        val = np.exp(-((pos - 0.5) * 4)**2) * 0.5 + 0.2
        center_pattern.append(max(0.0, val))
    initializations.append(center_pattern)

    # 5. Sparse peaks
    sparse_pattern = np.zeros(n_steps)
    positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    for pos in positions:
        sparse_pattern += np.exp(-((x - pos)**2) / 0.02) * 1.5
    initializations.append(sparse_pattern.tolist())

    return initializations

def adaptive_evolutionary_search(n_steps: int, max_time: float = 85.0) -> List[float]:
    """
    Multi-phase evolutionary optimization with adaptive parameters
    """
    start_time = time.time()

    # Create diverse initial population
    initializations = create_multiple_initializations(n_steps)

    # Evaluate all initializations
    best_score = -np.inf
    best_individual = None

    for init in initializations:
        try:
            score = evaluate_c2(init)
            if score > best_score:
                best_score = score
                best_individual = init.copy()
        except:
            continue

    # Use the best initialization as starting point
    if best_individual is None:
        # Fallback to basic initialization
        best_individual = [1.0] * n_steps

    # Phase 1: Global search with differential evolution (coarse)
    def objective(f_vals):
        return -evaluate_c2(np.array(f_vals))

    bounds = [(0.0, 2.0) for _ in range(n_steps)]

    # Try different evolutionary parameters
    params_list = [
        {'maxiter': 30, 'popsize': 15, 'mutation': (0.5, 1.0)},
        {'maxiter': 50, 'popsize': 25, 'mutation': (0.6, 1.0)},
        {'maxiter': 20, 'popsize': 10, 'mutation': (0.8, 1.0)}
    ]

    for params in params_list:
        if time.time() - start_time > max_time - 5:
            break

        try:
            result = differential_evolution(
                objective,
                bounds,
                seed=int(time.time()),
                maxiter=params['maxiter'],
                popsize=params['popsize'],
                mutation=params['mutation'],
                recombination=0.7,
                tol=1e-6,
                disp=False
            )

            current_score = evaluate_c2(result.x)
            if current_score > best_score:
                best_score = current_score
                best_individual = result.x.tolist()

        except:
            continue

    # Phase 2: Refinement with multiple techniques
    if time.time() - start_time < max_time - 3:
        # Try L-BFGS-B
        try:
            refined_result = minimize(
                objective,
                best_individual,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50}
            )
            current_score = evaluate_c2(refined_result.x)
            if current_score > best_score:
                best_score = current_score
                best_individual = refined_result.x.tolist()
        except:
            pass

        # Try Nelder-Mead as backup
        try:
            nm_result = minimize(
                objective,
                best_individual,
                method='Nelder-Mead',
                options={'maxiter': 30}
            )
            current_score = evaluate_c2(nm_result.x)
            if current_score > best_score:
                best_score = current_score
                best_individual = nm_result.x.tolist()
        except:
            pass

    return best_individual

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value using enhanced hybrid optimization approach."""
    # Use a larger number of steps for better resolution
    n_steps = 2000

    # Use adaptive evolutionary search
    try:
        final_values = adaptive_evolutionary_search(n_steps, max_time=85.0)
    except Exception:
        # Fallback to simple approach if anything fails
        final_values = [1.0] * n_steps

    # Post-processing: ensure non-negative values and normalize
    final_values = np.clip(final_values, 0, None)
    total = np.sum(final_values)
    if total > 0:
        final_values = final_values / total * 2.0

    # Convert to list of floats
    return [float(x) for x in final_values]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")