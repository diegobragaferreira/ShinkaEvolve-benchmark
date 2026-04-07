# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import block_diag
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def compute_convolution(seq):
    """Compute the convolution of a sequence with itself."""
    n = len(seq)
    conv = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            conv[i + j] += seq[i] * seq[j]
    return conv[:n]

def compute_c1(sequence):
    """Compute the C1 value for a given sequence."""
    n = len(sequence)
    if n < 1:
        return float('inf')
    
    sum_a = np.sum(sequence)
    if sum_a < 1e-10:
        return float('inf')
    
    conv = compute_convolution(sequence)
    max_conv = np.max(conv)
    
    c1 = 2 * n * max_conv / (sum_a ** 2)
    return c1

def evaluate_sequence(sequence):
    """Evaluate a sequence by computing its inverse C1."""
    c1 = compute_c1(sequence)
    if c1 == float('inf'):
        return 0.0  # Invalid sequence gets low score
    return 1.0 / c1  # Higher inverse C1 is better

def compute_hessian_matrix(n, sequence):
    """Compute the Hessian matrix for quadratic programming formulation."""
    # Placeholder for actual Hessian computation; simplified for now
    return np.eye(n)

def solve_quadratic_programming(sequence, max_iter=100):
    """Solve the quadratic programming problem for optimizing the sequence."""
    n = len(sequence)
    if n < 1:
        return sequence

    # Objective function: minimize -inverse_C1 = -1/C1
    # This is equivalent to maximizing C1, which means minimizing 1/C1
    def objective(x):
        # Reconstruct the sequence from the flattened input
        seq = np.array(x)
        sum_seq = np.sum(seq)
        if sum_seq < 1e-10:
            return float('inf')
        conv = compute_convolution(seq)
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return float('inf')
        return -(sum_seq ** 2) / (2 * n * max_conv)

    # Constraints
    # Non-negativity
    bounds = [(0, 1000) for _ in range(n)]
    
    # Sum constraint (optional: normalize sum to 1 for numerical stability)
    def sum_constraint(x):
        return np.sum(x) - 1.0  # Normalize sum to 1
    
    # Initial guess
    x0 = np.array(sequence)
    
    # SLSQP solver is suitable for constrained problems
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints={'type': 'eq', 'fun': sum_constraint},
                       options={'maxiter': max_iter})
        if res.success:
            return res.x.tolist()
        else:
            return sequence  # Return original if optimization fails
    except Exception as e:
        warnings.warn(f"Optimization error: {str(e)}")
        return sequence

def construct_optimal_step_function(n):
    """Construct an initial optimized step function based on known properties."""
    # Use an exponential decay pattern common in similar problems
    seq = np.exp(-np.linspace(0, 3, n))
    
    # Normalize to ensure sum is reasonable (not too small)
    sum_seq = np.sum(seq)
    if sum_seq > 0:
        seq = seq / sum_seq * 100  # Scale up for better numerical behavior
    else:
        seq = np.ones(n) * 100 / n  # Fallback to uniform
        
    # Clip to reasonable bounds
    seq = np.clip(seq, 0, 1000)
    
    return seq.tolist()

def adaptive_length_selection(current_len, fitness_history, patience=5):
    """Adaptively select sequence length based on performance trends."""
    if len(fitness_history) < patience + 1:
        return current_len
    
    recent_changes = [
        fitness_history[-i] - fitness_history[-i-1]
        for i in range(1, min(patience, len(fitness_history)-1))
    ]
    
    avg_change = np.mean(recent_changes) if recent_changes else 0
    
    if avg_change > 0.001:
        new_len = min(current_len + 50, 1500)
    elif avg_change < -0.001 and current_len > 50:
        new_len = max(current_len - 50, 50)
    else:
        new_len = current_len
    
    return new_len

def search_for_best_sequence() -> list[float]:
    """Main search function using quadratic programming approach."""
    start_time = time.time()
    max_time = 170  # Maximum execution time in seconds
    best_sequence = None
    best_fitness = 0.0
    fitness_history = []
    
    # Multi-start approach with varied initial conditions
    num_starts = 10
    for start_idx in range(num_starts):
        if time.time() - start_time > max_time:
            break
            
        # Vary sequence length and initial pattern
        n = np.random.randint(50, 800)
        current_sequence = construct_optimal_step_function(n)
        
        # Iterative optimization within time budget
        iter_count = 0
        while time.time() - start_time < max_time and iter_count < 40:
            # Quadratic programming optimization
            optimized_sequence = solve_quadratic_programming(current_sequence)
            
            # Evaluate the new sequence
            current_fitness = evaluate_sequence(optimized_sequence)
            fitness_history.append(current_fitness)
            
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_sequence = optimized_sequence[:]
            
            # Update sequence with optimized version
            current_sequence = optimized_sequence
            
            # Occasionally adjust sequence length based on performance
            if iter_count % 10 == 0:
                n = adaptive_length_selection(n, fitness_history)
                if n != len(current_sequence):
                    # Adjust length by padding or truncating
                    if n > len(current_sequence):
                        current_sequence.extend([0.0] * (n - len(current_sequence)))
                    else:
                        current_sequence = current_sequence[:n]
                        
            iter_count += 1
            
    # Final validation and fallback
    if best_sequence is None:
        n = np.random.randint(50, 800)
        best_sequence = construct_optimal_step_function(n)
        
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")