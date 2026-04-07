# EVOLVE-BLOCK-START
import numpy as np
import nevergrad as ng
from scipy.signal import fftconvolve
from scipy.optimize import minimize
import time

def convolve_fft(a):
    """Compute convolution using FFT for efficiency."""
    n = len(a)
    # Pad to avoid circular convolution effects
    padded_len = 2 * n - 1
    a_padded = np.pad(a, (0, padded_len - n), mode='constant')
    b_padded = np.pad(a, (0, padded_len - n), mode='constant')
    result = fftconvolve(a_padded, b_padded, mode='valid')
    return result[:padded_len]

def compute_c1(sequence):
    """Compute C1 constant from the given sequence."""
    n = len(sequence)
    if n == 0:
        return float('inf')
    
    sum_a = np.sum(sequence)
    if sum_a < 1e-10:
        return float('inf')
    
    # Compute convolution
    conv_result = convolve_fft(sequence)
    max_conv = np.max(conv_result)
    
    # Compute C1
    c1 = 2 * n * max_conv / (sum_a ** 2)
    return c1

def eval_function(x):
    """Evaluation function for nevergrad optimization."""
    # Ensure non-negativity and bounded heights
    x = np.clip(x, 0, 1000)
    if np.sum(x) < 0.01:
        return float('inf')
    c1 = compute_c1(x)
    # Return inverse to maximize 1/C1 (minimize C1)
    return 1.0 / c1 if c1 > 0 else float('inf')

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence using nevergrad optimization."""
    n = len(sequence)
    if n == 0:
        return None
    
    # Set up nevergrad optimizer with a suitable search space
    instrum = ng.p.Array(shape=(n,), bounds=(0, 1000))
    optimizer = ng.optimizers.OnePlusOne(instrumentation=instrum, budget=50)
    
    # Initialize with current sequence
    try:
        # Create a good starting point by perturbing the current sequence
        current_x = np.array(sequence)
        # Add small noise to encourage exploration
        noise = np.random.normal(0, 0.1, n)
        initial_guess = np.clip(current_x + noise, 0, 1000)
        
        # Evaluate and optimize
        for _ in range(10):  # Limited iterations for speed
            candidate = optimizer.ask()
            value = eval_function(candidate.args[0])
            optimizer.tell(candidate, value)
        
        # Get the best candidate
        best_candidate = optimizer.provide_recommendation()
        new_sequence = best_candidate.args[0]
        
        # Ensure all values are non-negative and reasonably bounded
        new_sequence = np.clip(new_sequence, 0, 1000)
        if np.sum(new_sequence) < 0.01:
            return None
        
        return new_sequence.tolist()
    except Exception as e:
        # Fallback to simple adaptive mutation if nevergrad fails
        try:
            # Simple gradient ascent with adaptive step size
            t = min(0.05, 0.01 + 0.01 * np.log(n + 1))
            new_sequence = [(1 - t) * x + t * max(x, 1e-6) for x in sequence]
            return new_sequence
        except:
            return None

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence using nevergrad."""
    # Start with a simple initialization
    n = 50  # Initial sequence length
    sequence = [np.random.uniform(0.1, 10.0) for _ in range(n)]
    
    # Optimization loop
    max_iterations = 30
    for iteration in range(max_iterations):
        updated_sequence = get_good_direction_to_move_into(sequence)
        if updated_sequence is not None:
            sequence = updated_sequence
        else:
            # Fallback mutation if optimization fails
            index = np.random.randint(len(sequence))
            sequence[index] = max(0.01, sequence[index] * np.random.normal(1, 0.1))
    
    # Final refinement with local optimization if needed
    try:
        # Refine with scipy minimize
        def objective(x):
            return -eval_function(x)  # Minimize negative to maximize
        
        # Ensure non-negativity and reasonable bounds
        bounds = [(0, 1000) for _ in range(len(sequence))]
        result = minimize(objective, sequence, bounds=bounds, method='L-BFGS-B', options={'maxiter': 20})
        if result.success:
            refined_sequence = result.x
            refined_sequence = np.clip(refined_sequence, 0, 1000)
            if np.sum(refined_sequence) >= 0.01:
                sequence = refined_sequence.tolist()
    except:
        pass  # Continue with existing sequence if refinement fails
    
    return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")