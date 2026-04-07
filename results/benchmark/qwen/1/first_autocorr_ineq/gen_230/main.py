# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
from typing import List, Optional
import random
import time
from scipy.optimize import minimize
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import optuna

# Set a fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autocorrelation_constant(sequence: List[float]) -> tuple[float, float]:
    """Computes the autocorrelation constant C1 and its reciprocal 1/C1."""
    if not sequence or sum(sequence) < 0.01:
        return (float('inf'), 0.0)

    n = len(sequence)
    # Use FFT-based convolution for efficiency O(n log n)
    conv = np.convolve(sequence, sequence, mode='full')
    max_conv = np.max(conv)
    sum_seq = sum(sequence)

    if sum_seq == 0:
        return (float('inf'), 0.0)

    c1 = 2 * n * max_conv / (sum_seq ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return (c1, inv_c1)

def generate_peak_shaped_sequence(length: int) -> List[float]:
    """Generate a sequence designed to minimize convolution peaks."""
    # Create a decreasing sequence to reduce autocorrelation peaks
    sequence = []
    for i in range(length):
        # Use exponential decay to reduce later values
        base_val = 100.0 * np.exp(-i * 0.02)
        # Add a slight oscillation to introduce some variation without creating peaks
        oscillation = 5.0 * np.sin(i * 0.1)
        val = max(0.01, base_val + oscillation)
        sequence.append(val)
    return sequence

def generate_focused_step_sequence(length: int) -> List[float]:
    """Generate a focused step sequence with fewer, larger steps."""
    sequence = [0.0] * length
    num_steps = max(2, min(8, length // 50))  # Fewer steps for longer sequences
    step_positions = []
    for i in range(num_steps):
        pos = int((i + 1) * length / (num_steps + 1))
        step_positions.append(pos)
    step_positions.sort()
    
    # Assign heights with a focus on reducing early peaks
    for i, pos in enumerate(step_positions):
        if i < len(step_positions) - 1:
            end_pos = step_positions[i+1]
        else:
            end_pos = length
        # Heights decrease to reduce convolution peaks
        height = 100.0 * (1.0 - i / (num_steps - 1)) if num_steps > 1 else 100.0
        sequence[pos:end_pos] = [height] * (end_pos - pos)
    return sequence

def local_search_improvement(sequence: List[float], max_iter: int = 50) -> List[float]:
    """
    Apply local search improvement using gradient-free techniques.
    """
    # Convert to numpy array for easier manipulation
    x0 = np.array(sequence)
    n = len(x0)

    # Define objective function to minimize (negative of 1/C1)
    def objective(x):
        # Ensure non-negativity and avoid near-zero values
        x = np.maximum(x, 1e-6)
        c1, _ = compute_autocorrelation_constant(x.tolist())
        return -1.0 / c1 if c1 > 0 else 1e6

    # Use Nelder-Mead for gradient-free optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': max_iter, 'initial_simplex': None})
        if result.success:
            refined = np.maximum(result.x, 1e-6).tolist()
            return refined
    except Exception as e:
        pass  # Fall back to original if optimization fails

    return sequence

def bayesian_optimization_approach(max_evals: int = 100) -> List[float]:
    """
    Uses Bayesian optimization to find a good sequence.
    This approach is more efficient than exhaustive exploration.
    """
    def objective_function(params):
        # Convert parameters to sequence
        length = int(params[0])
        params = params[1:]
        # Rescale parameters to [0.01, 100.0] range
        scaled_params = 0.01 + 99.99 * np.array(params) / np.max(params) if np.max(params) > 0 else params
        sequence = scaled_params.tolist()
        
        # Pad or truncate to desired length
        if len(sequence) > length:
            sequence = sequence[:length]
        elif len(sequence) < length:
            sequence.extend([scaled_params[-1]] * (length - len(sequence)))
            
        c1, inv_c1 = compute_autocorrelation_constant(sequence)
        return -inv_c1  # Minimize negative inverse for maximization

    # Start with a good initial sequence
    best_sequence = generate_peak_shaped_sequence(200)
    best_inv_c1 = compute_autocorrelation_constant(best_sequence)[1]
    
    # Define bounds for optimization (length, parameters)
    bounds = [(100, 1000)]  # Length between 100 and 1000
    # Parameters can be up to 1000 (for simplicity)
    bounds.extend([(0.0, 1.0)] * 200)  # Up to 200 parameters

    # Optimize using Optuna
    try:
        study = optuna.create_study(direction='maximize')
        for _ in range(max_evals):
            # Sample random point within bounds
            sample = [random.uniform(b[0], b[1]) for b in bounds]
            # Use the sample as the starting point for local search
            try:
                # Create a sequence from sampled parameters
                length = int(sample[0])
                seq_params = sample[1:length+1] if length+1 <= len(sample) else sample[1:]
                sequence = [abs(100 * p) for p in seq_params]
                # Pad or truncate to exact length
                if len(sequence) > length:
                    sequence = sequence[:length]
                elif len(sequence) < length:
                    sequence.extend([100.0] * (length - len(sequence)))
                
                # Evaluate
                c1, inv_c1 = compute_autocorrelation_constant(sequence)
                study.tell(study.ask(), -inv_c1)
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = sequence.copy()
            except Exception:
                continue
                
        # Final local refinement
        refined = local_search_improvement(best_sequence)
        _, final_inv_c1 = compute_autocorrelation_constant(refined)
        if final_inv_c1 > best_inv_c1:
            best_sequence = refined
            
    except Exception as e:
        pass  # Fallback to default
        
    return best_sequence

def search_for_best_sequence() -> list[float]:
    """Main function to find the best coefficient sequence using probabilistic optimization."""
    # Try Bayesian optimization approach
    try:
        sequence = bayesian_optimization_approach(max_evals=80)
    except Exception:
        # Fallback to a structured approach
        sequence = generate_peak_shaped_sequence(200)
    
    # Apply local search improvement
    refined = local_search_improvement(sequence)
    _, inv_c1 = compute_autocorrelation_constant(refined)
    
    # If not improving much, fall back to a step-based sequence
    if inv_c1 < 0.6:
        sequence = generate_focused_step_sequence(200)
        refined = local_search_improvement(sequence)
        _, inv_c1 = compute_autocorrelation_constant(refined)
    
    return refined

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")