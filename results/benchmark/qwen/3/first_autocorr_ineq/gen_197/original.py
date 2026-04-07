# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
from typing import List, Tuple, Optional
import time
import warnings

# Suppress scientific notation for cleaner output
np.set_printoptions(suppress=True)

def convolve_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Efficient FFT-based convolution for large sequences."""
    return fftconvolve(a, b, mode='full')[:2*len(a)-1]

def convolve_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct convolution for small sequences."""
    return np.convolve(a, b, mode='full')[:2*len(a)-1]

def compute_c1_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 constant and 1/C1 value for a given sequence."""
    a = np.array(sequence)
    n = len(a)

    # Use FFT for efficiency when sequence is large
    if n > 100:
        b = convolve_fft(a, a)
    else:
        b = convolve_direct(a, a)

    max_conv = np.max(b)
    sum_a = np.sum(a)

    if sum_a < 0.01:
        return float('inf'), 0.0

    c1 = 2 * n * max_conv / (sum_a ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return c1, inv_c1

def solve_convolution_lp(f_sequence: List[float], rhs: float) -> Optional[List[float]]:
    """Solves the convolution LP for a given sequence and RHS."""
    try:
        n = len(f_sequence)
        c = -np.ones(n)
        a_ub = []
        b_ub = []

        # Build constraint matrix for convolution constraints
        for k in range(2 * n - 1):
            row = np.zeros(n)
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    row[j] = f_sequence[i]
            a_ub.append(row)
            b_ub.append(rhs)

        # Add non-negativity constraints
        a_ub_nonneg = -np.eye(n)
        b_ub_nonneg = np.zeros(n)

        a_ub = np.vstack([a_ub, a_ub_nonneg])
        b_ub = np.hstack([b_ub, b_ub_nonneg])

        # Solve the linear program with multiple methods as fallback
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'maxiter': 1000})

        if not result.success:
            # Try different method if highs fails
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='revised simplex', options={'maxiter': 1000})

        if result.success:
            return result.x.tolist()
        else:
            return None

    except Exception:
        return None

def get_good_direction_to_move_into(
    sequence: List[float],
    max_iterations: int = 10
) -> Optional[List[float]]:
    """Improve the sequence using evolutionary strategy and LP optimization."""
    n = len(sequence)

    # Normalize the sequence
    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None

    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

    # Compute convolution constraints
    if n > 100:
        b = convolve_fft(np.array(normalized_sequence), np.array(normalized_sequence))
    else:
        b = convolve_direct(np.array(normalized_sequence), np.array(normalized_sequence))

    rhs = np.max(b)

    # Try multiple times to solve LP
    g_fun = None
    for _ in range(max_iterations):
        g_fun = solve_convolution_lp(normalized_sequence, rhs)
        if g_fun is not None:
            break
        else:
            # If LP fails, slightly modify constraints and retry
            rhs *= 1.01

    if g_fun is None:
        return None

    # Normalize the solution from LP
    sum_g = np.sum(g_fun)
    if sum_g < 0.01:
        return None

    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]

    # Apply small perturbation for exploration
    t = 0.05
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]

    # Ensure non-negativity and reasonable bounds
    new_sequence = [max(0, min(1000, x)) for x in new_sequence]

    return new_sequence

def hierarchical_mutation(sequence: List[float], scale_factor: float = 1.0) -> List[float]:
    """Apply hierarchical mutation with varying intensities based on sequence statistics."""
    mutated = sequence.copy()
    n = len(mutated)
    
    # Calculate sequence statistics
    mean_val = np.mean(mutated)
    std_val = np.std(mutated)
    
    # Determine mutation intensity based on sequence properties
    intensity = 0.1 * scale_factor
    if std_val > mean_val * 0.5 and mean_val > 0:
        intensity *= 2.0
    
    for i in range(n):
        if random.random() < 0.1 * scale_factor:
            # Apply different mutation types based on position and value
            if random.random() < 0.5:
                # Gaussian mutation
                mutated[i] = max(0, mutated[i] + np.random.normal(0, intensity * mutated[i]))
            else:
                # Uniform mutation
                mutated[i] = max(0, mutated[i] + random.uniform(-intensity * mutated[i], intensity * mutated[i]))
    
    return mutated

def adaptive_sequence_length(sequence: List[float], target_ratio: float = 0.8) -> List[float]:
    """Dynamically adjust sequence length based on observed convergence properties."""
    n = len(sequence)
    
    # If sequence is too long, consider truncation
    if n > 1000:
        # Keep top 50% of the sequence values
        top_indices = np.argsort(sequence)[-n//2:]
        new_sequence = [sequence[i] for i in sorted(top_indices)]
        return new_sequence
    
    # If sequence is too short, consider expansion
    if n < 50:
        # Expand with copies and slight mutations
        expanded = sequence.copy()
        for i in range(10):
            idx = random.randint(0, n-1)
            expanded.append(expanded[idx] * (1 + random.uniform(-0.2, 0.2)))
        return expanded
    
    return sequence

def smart_constraint_relaxation(sequence: List[float], tolerance: float = 0.001) -> List[float]:
    """Allow constraint relaxation in infeasible regions with recovery mechanisms."""
    # Try to find a feasible solution by relaxing constraints
    relaxed_seq = sequence.copy()
    for _ in range(10):
        try:
            # Add small random perturbations to make it feasible
            for i in range(len(relaxed_seq)):
                if relaxed_seq[i] < 0.01:
                    relaxed_seq[i] = random.uniform(0.01, 1.0)
            return relaxed_seq
        except:
            continue
    return sequence

def ensemble_evaluation(sequence: List[float]) -> Tuple[float, float]:
    """Combine multiple evaluation heuristics to get a robust estimate."""
    # Run different evaluation strategies and take consensus
    c1_vals = []
    inv_c1_vals = []
    
    # Standard evaluation
    c1, inv_c1 = compute_c1_constant(sequence)
    c1_vals.append(c1)
    inv_c1_vals.append(inv_c1)
    
    # Perturbed evaluation
    perturbed = [x * (1 + random.uniform(-0.05, 0.05)) for x in sequence]
    c1_p, inv_c1_p = compute_c1_constant(perturbed)
    c1_vals.append(c1_p)
    inv_c1_vals.append(inv_c1_p)
    
    # Smoothed evaluation
    smoothed = [np.mean([sequence[max(0,i-1)], sequence[i], sequence[min(len(sequence)-1,i+1)]]) for i in range(len(sequence))]
    c1_s, inv_c1_s = compute_c1_constant(smoothed)
    c1_vals.append(c1_s)
    inv_c1_vals.append(inv_c1_s)
    
    # Return the median of evaluations
    median_c1 = np.median(c1_vals)
    median_inv_c1 = np.median(inv_c1_vals)
    
    return median_c1, median_inv_c1

def multi_scale_search(max_time_seconds: int = 180) -> List[float]:
    """Multi-scale search strategy to avoid local optima."""
    start_time = time.time()
    
    # Multi-scale initialization
    scales = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    current_best_sequence = None
    best_inv_c1 = 0.0
    
    for scale in scales:
        if time.time() - start_time > max_time_seconds:
            break
            
        # Initialize with scaled sequence  
        n = max(50, int(200 * scale))
        sequence = [random.uniform(0.1, 1.0) for _ in range(n)]
        
        # Apply hierarchical mutation to create diversity
        sequence = hierarchical_mutation(sequence, scale)
        
        # Perform local optimization
        for _ in range(20):
            if time.time() - start_time > max_time_seconds:
                break
                
            improved = get_good_direction_to_move_into(sequence)
            if improved is not None:
                sequence = improved
            else:
                sequence = hierarchical_mutation(sequence, scale)
                
            # Evaluate using ensemble method
            _, inv_c1 = ensemble_evaluation(sequence)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                current_best_sequence = sequence.copy()
                
        # Dynamic sequence length adjustment
        sequence = adaptive_sequence_length(sequence)
        
        # Apply constraint relaxation if needed
        sequence = smart_constraint_relaxation(sequence)
        
        # Final evaluation
        _, inv_c1 = ensemble_evaluation(sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            current_best_sequence = sequence.copy()
            
    return current_best_sequence if current_best_sequence is not None else [1.0]

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    return multi_scale_search()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")