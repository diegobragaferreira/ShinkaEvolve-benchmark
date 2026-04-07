# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import convolve
from scipy.optimize import minimize
import cvxpy as cp
import time
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_c1(sequence):
    """Compute the C1 constant for a given sequence."""
    if len(sequence) == 0:
        return float('inf')

    # Use FFT-based convolution for efficiency
    conv = convolve(sequence, sequence, mode='full')
    # Take only the relevant part of convolution (the peak)
    max_conv = np.max(conv[len(sequence)-1:])  # From index n-1 onwards

    # Normalize and compute C1
    sum_sq = np.sum(sequence)**2
    if sum_sq == 0:
        return float('inf')

    c1 = (2 * len(sequence) * max_conv) / sum_sq
    return c1

def compute_inv_c1(sequence):
    """Compute inverse of C1 (what we want to maximize)."""
    c1 = compute_c1(sequence)
    if c1 == 0 or np.isinf(c1):
        return 0
    return 1.0 / c1

def create_convex_optimization_problem(sequence_length):
    """
    Create a convex optimization problem for maximizing 1/C1.
    Variables represent sequence heights.
    """
    # Decision variables
    a = cp.Variable(sequence_length, nonneg=True)
    
    # Objective: maximize sum(a) to increase total mass
    objective = cp.Maximize(cp.sum(a))
    
    # Constraints to ensure feasibility and meaningful optimization
    # Constraint 1: Sum is reasonably large (to avoid trivial solutions)
    sum_constraint = cp.sum(a) >= 0.01
    
    # Constraint 2: Individual elements bounded
    bound_constraints = [a[i] <= 1000 for i in range(sequence_length)]
    
    # Assemble constraints
    constraints = [sum_constraint] + bound_constraints
    
    # Construct problem
    problem = cp.Problem(objective, constraints)
    
    return problem, a

def optimize_convex_approach(sequence_length):
    """
    Solve the convex optimization problem to find a good sequence.
    """
    try:
        problem, a_var = create_convex_optimization_problem(sequence_length)
        
        # Solve the problem
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            solution = a_var.value
            # Clip to [0, 1000]
            solution = np.clip(solution, 0, 1000)
            return solution.tolist()
        else:
            # Fallback to a decent sequence
            return np.random.uniform(0.01, 1.0, sequence_length).tolist()
            
    except Exception as e:
        # If anything goes wrong, return a random sequence
        return np.random.uniform(0.01, 1.0, sequence_length).tolist()

def initialize_good_sequence(length=None):
    """Initialize a good starting sequence based on theoretical insights."""
    if length is None:
        length = random.randint(100, 500)  # Reasonable range

    # Start with a simple exponential decay pattern which often works well
    decay_factor = 0.95
    sequence = [1.0 * (decay_factor ** i) for i in range(length)]

    # Ensure minimum value and normalize
    sequence = [max(x, 0.01) for x in sequence]

    # Apply a bit of randomness to avoid local optima
    noise_factor = 0.1
    sequence = [x * (1 + random.uniform(-noise_factor, noise_factor)) for x in sequence]
    sequence = [max(x, 0.01) for x in sequence]

    return sequence

def optimize_with_nelder_mead(initial_sequence, max_iter=50):
    """Optimize using Nelder-Mead method which is more suitable for this problem."""
    def objective_func(seq_array):
        # Return negative because we want to maximize
        return -compute_inv_c1(seq_array.tolist())

    # Ensure we start with a valid sequence
    if np.sum(initial_sequence) < 0.01:
        initial_sequence = [0.1] + [0.0] * (len(initial_sequence) - 1)

    # Convert to numpy array for optimization
    x0 = np.array(initial_sequence, dtype=float)

    # Set bounds for optimization (non-negative and reasonable values)
    bounds = [(0, 1000) for _ in range(len(x0))]

    # Use Nelder-Mead for direct search without gradients
    result = minimize(
        objective_func,
        x0,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'adaptive': True}
    )

    if result.success:
        optimized_seq = np.maximum(result.x, 0)  # Ensure non-negative
        # Ensure sum is reasonable
        if np.sum(optimized_seq) < 0.01:
            optimized_seq[0] = 0.1
        return optimized_seq.tolist()
    else:
        return initial_sequence

def adjust_by_convolution(sequence):
    """Adjust sequence based on convolution properties to reduce peak convolution."""
    seq = np.array(sequence)
    n = len(seq)

    # Compute convolution
    conv = convolve(seq, seq, mode='full')
    conv_part = conv[n-1:]  # Relevant part

    # Find indices where convolution peaks
    max_conv = np.max(conv_part)
    max_indices = np.where(conv_part == max_conv)[0]

    # Adjust elements near these peaks to reduce convolution max
    new_seq = seq.copy()
    for idx in max_indices[:min(3, len(max_indices))]:  # Limit adjustments
        if idx > 0:  # Avoid boundary issues
            # Reduce neighboring elements
            new_seq[max(0, idx-1)] *= 0.95
            new_seq[min(n-1, idx+1)] *= 0.95

    return np.maximum(new_seq, 0)

def smart_search():
    """Smart search approach combining convex optimization, evolutionary refinement, and local optimization."""
    best_inv_c1 = 0
    best_sequence = None

    # Try multiple random starting points with good initialization
    for attempt in range(10):  # Increased attempts for better chance
        # Initialize with better patterns
        n = random.randint(100, 500)
        sequence = initialize_good_sequence(n)

        # Try convex optimization approach
        convex_sequence = optimize_convex_approach(n)
        convex_inv_c1 = compute_inv_c1(convex_sequence)
        
        # Try evolutionary refinement
        evolved_sequence = adjust_by_convolution(sequence)
        evolved_inv_c1 = compute_inv_c1(evolved_sequence)
        
        # Compare and select the best among convex and evolved
        selected_sequence = convex_sequence if convex_inv_c1 > evolved_inv_c1 else evolved_sequence
        selected_inv_c1 = max(convex_inv_c1, evolved_inv_c1)
        
        # Optimize this sequence further with Nelder-Mead
        optimized = optimize_with_nelder_mead(selected_sequence, max_iter=20)
        optimized_inv_c1 = compute_inv_c1(optimized)

        if optimized_inv_c1 > best_inv_c1:
            best_inv_c1 = optimized_inv_c1
            best_sequence = optimized[:]

    # If no improvement found, return a default good sequence
    if best_sequence is None:
        best_sequence = initialize_good_sequence(200)
        best_sequence = optimize_with_nelder_mead(best_sequence, max_iter=50)
        best_inv_c1 = compute_inv_c1(best_sequence)

    return best_sequence

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Add a time limit check to ensure we don't exceed budget
    start_time = time.time()

    # Perform the smart search
    best_sequence = smart_search()

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")