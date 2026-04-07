# EVOLVE-BLOCK-START
import numpy as np
import nevergrad as ng
from scipy.signal import fftconvolve
import time

def compute_c1_constant(sequence):
    """
    Computes the C1 constant for a given sequence.
    C1 = 2n * max(convolution) / (sum(sequence))^2
    We aim to maximize 1/C1, which is equivalent to minimizing C1.
    """
    n = len(sequence)
    if n == 0:
        return float('inf')
    
    sum_seq = np.sum(sequence)
    if sum_seq < 1e-10:
        return float('inf')

    # Use FFT for efficiency, especially for longer sequences
    if n > 100:
        conv_result = fftconvolve(sequence, sequence, mode='full')
        conv_result = conv_result[:2*n-1]
    else:
        conv_result = np.convolve(sequence, sequence)
        
    max_conv = np.max(conv_result)
    
    c1 = (2 * n * max_conv) / (sum_seq ** 2)
    return c1

def evaluate_sequence(sequence):
    """
    Evaluates a sequence by returning the inverse of C1 constant.
    Higher values are better (we maximize 1/C1).
    """
    c1 = compute_c1_constant(sequence)
    if np.isinf(c1) or np.isnan(c1):
        return -float('inf')  # Penalize invalid sequences
    return 1.0 / c1 if c1 > 0 else -float('inf')

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Use nevergrad optimizer to get a potentially better sequence."""
    start_time = time.time()
    
    n = len(sequence)
    if n < 1:
        return None
    
    # Set up the optimization problem with nevergrad
    # Define the dimensionality of the optimization space
    dimension = n
    
    # Define the search space: each element is a non-negative real number
    # with a reasonable upper bound (clipped to 1000 per requirements)
    bounds = [[0.0, 1000.0]] * dimension
    instrum = ng.p.Array(shape=(dimension,), bounds=bounds)

    # Define the objective function to minimize (negative of our metric to maximize)
    def objective(x):
        # Clip values to valid range
        x = np.clip(x, 0, 1000)
        # Compute 1/C1 (our metric to maximize)
        inv_c1 = evaluate_sequence(x)
        return -inv_c1  # Negative because we're minimizing

    # Create optimizer with a suitable algorithm (Bayesian optimization-like)
    optimizer = ng.optimizers.CMA(options={"popsize": 10, "ftarget": -1e30})  # Use CMA-ES
    optimizer.register_evaluation(instrum, objective)

    # Optimization loop with time limit
    max_time = 0.1  # Limit evaluation time
    try:
        # Suggest new candidate and evaluate (this is a simplified version)
        # In practice, we'd do multiple iterations, but let's make it fast
        # Use a simple population-based approach for quick improvement
        candidates = []
        for _ in range(5):  # Generate some candidates
            cand = optimizer.ask()
            candidates.append(cand.value)
            
        # Evaluate all candidates
        best_candidate = None
        best_score = -float('inf')
        
        for cand in candidates:
            # Ensure it meets constraints
            if np.sum(cand) > 0.01:
                score = evaluate_sequence(cand)
                if score > best_score:
                    best_score = score
                    best_candidate = cand
                    
        if best_candidate is not None:
            # Return a slight perturbation to avoid getting stuck
            # But since we want the best, just return it
            return best_candidate.tolist()
        else:
            # Fallback to original sequence if no improvement found
            return sequence
            
    except Exception as e:
        # On any exception, fallback gracefully
        return sequence

def search_for_best_sequence() -> list[float]:
    """Searches for the best sequence using nevergrad-based optimization."""
    # Start with a good initialization strategy
    np.random.seed(42)
    
    # Try a mix of strategies for different starting points
    strategies = [
        # Uniform random
        np.random.rand(50).tolist(),
        # Exponential
        [np.random.exponential(1.0) for _ in range(50)],
        # Gamma
        [np.random.gamma(2.0, 1.0) for _ in range(50)],
        # Some structured to encourage exploration
        [1.0 if i % 3 == 0 else 0.1 for i in range(50)]
    ]
    
    # Choose initial best based on performance
    best_sequence = None
    best_score = -float('inf')
    
    for strategy in strategies:
        # Ensure minimum length and positivity
        strategy = [max(x, 0.01) for x in strategy]
        if len(strategy) < 10:
            strategy.extend([0.1] * (10 - len(strategy)))
        score = evaluate_sequence(strategy)
        if score > best_score:
            best_score = score
            best_sequence = strategy.copy()
    
    # Now refine using nevergrad
    refined_sequence = get_good_direction_to_move_into(best_sequence)
    
    # Final check
    final_score = evaluate_sequence(refined_sequence)
    if final_score > best_score:
        return refined_sequence
    else:
        return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")