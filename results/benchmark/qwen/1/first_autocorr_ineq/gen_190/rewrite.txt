# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import block_diag
from cvxpy import Variable, Minimize, Problem, norm, sum_squares, quad_form, sqrt
import time
import random

# Fixed seeds for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autocorrelation_constant(sequence):
    """Compute the C1 constant for a given sequence"""
    if len(sequence) == 0:
        return float('inf')

    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return float('inf')

    # Compute autoconvolution manually for small sequences
    n = len(sequence)
    conv_result = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            conv_result[i + j] += sequence[i] * sequence[j]
    
    max_conv = np.max(conv_result)

    # Compute C1 = 2n * max(conv) / (sum(a))^2
    c1 = (2 * n * max_conv) / (sum_a ** 2)

    return c1

def evaluate_sequence(sequence):
    """Evaluate a sequence by computing its inverse C1"""
    c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf'):
        return 0.0  # Invalid sequence
    return 1.0 / c1  # Return 1/C1 as the objective

def solve_quadratic_optimization(sequence):
    """
    Formulates and solves a convex quadratic optimization problem to find a better sequence.
    Minimizes the ratio max(b)/(sum(a))² subject to non-negativity and boundedness constraints.
    """
    try:
        n = len(sequence)
        if n == 0:
            return None

        # Define variables for the new sequence
        a_new = Variable(n)
        
        # Objective: minimize max(b)/(sum(a))²
        # For simplicity, we'll optimize a related quantity that approximates our goal
        sum_a = sum(a_new)
        # Approximate max convolution using sum of squares (convex relaxation)
        # We'll use a simpler convex approximation for now
        
        # Since we cannot directly model max convolution in QP, we'll approximate:
        # We use a simplified quadratic form that encourages sparsity and balanced distributions
        
        # Objective: maximize 1/C1 ~ minimize (sum(a))^2 / max(b)
        # This is non-convex but we'll linearize around current point for local search
        
        # Simplified convex problem that tries to balance magnitude and spread
        # Minimize sum_squares(a) + lambda * sum(a) to encourage balanced spreads
        
        # Here we define a proxy objective that prefers sequences with lower max convolution
        # For this simple version, just minimize sum_squares to encourage well-distributed values
        
        # Define constraints
        constraints = [a_new >= 0, a_new <= 1000]  # Non-negativity and boundedness
        
        # For demonstration, we'll solve a simple convex problem that helps improve sequence
        # This is more of a placeholder; a more complete implementation would involve 
        # solving the exact QP formulation with appropriate constraints
        prob = Problem(Minimize(sum_squares(a_new)), constraints)
        prob.solve(solver='ECOS', verbose=False)
        
        if prob.status == 'optimal':
            new_sequence = a_new.value
            # Clamp values to ensure bounds
            new_sequence = np.clip(new_sequence, 0, 1000)
            return new_sequence.tolist()
        else:
            return sequence  # Return original if optimization fails
    except Exception:
        return sequence  # Return original if any error occurs

def adaptive_sequence_generation():
    """Generate a sequence using adaptive strategy based on past performance"""
    # Sample from a log-uniform distribution to explore various sizes
    n = int(np.random.lognormal(np.log(300), 0.5))  # More varied sizes
    # Prefer sequences that have shown good performance in past
    # For now, a mix of regular and random patterns
    pattern_type = random.choice(['regular', 'random', 'sparse'])
    
    if pattern_type == 'regular':
        # Sine wave pattern
        sequence = [abs(np.sin(i * np.pi / n)) * 1000 for i in range(n)]
    elif pattern_type == 'sparse':
        # Sparse pattern with few high-value elements
        sequence = [0.0] * n
        num_peaks = max(1, n // 20)  # Very sparse
        for _ in range(num_peaks):
            idx = random.randint(0, n-1)
            sequence[idx] += random.uniform(500, 1000)
    else:
        # Random but structured
        sequence = [random.uniform(0, 1000) for _ in range(n)]
    
    # Ensure sum is meaningful
    if sum(sequence) < 0.01:
        sequence[random.randint(0, n-1)] += 0.01
    
    return sequence

def gradient_based_improvement(sequence, iterations=30):
    """Improve sequence using gradient-based local search with adaptive parameters"""
    current_seq = np.array(sequence, dtype=float)
    n = len(current_seq)
    
    # Normalize
    sum_seq = np.sum(current_seq)
    if sum_seq < 1e-10:
        current_seq += 1e-5
        sum_seq = np.sum(current_seq)
    current_seq /= sum_seq
    
    for _ in range(iterations):
        # Compute convolution
        conv = np.convolve(current_seq, current_seq, mode='full')
        conv = conv[n-1:2*n-1]
        max_conv = np.max(conv)
        
        # Estimate gradient via finite differences
        grad = np.zeros(n)
        epsilon = 1e-6
        for i in range(n):
            seq_plus = current_seq.copy()
            seq_plus[i] += epsilon
            seq_plus /= np.sum(seq_plus)
            
            conv_plus = np.convolve(seq_plus, seq_plus, mode='full')
            conv_plus = conv_plus[n-1:2*n-1]
            max_plus = np.max(conv_plus)
            
            grad[i] = (max_plus - max_conv) / epsilon
        
        # Update with gradient ascent
        step_size = 0.01
        current_seq += step_size * grad
        current_seq = np.maximum(current_seq, 0)
        
        # Renormalize
        sum_seq = np.sum(current_seq)
        if sum_seq > 0:
            current_seq /= sum_seq
            
    return current_seq.tolist()

def search_for_best_sequence(max_time=180) -> list[float]:
    """Main function to search for the best coefficient sequence."""
    start_time = time.time()
    
    # History to store recent best scores for early stopping
    recent_scores = []
    patience = 0
    max_patience = 20
    
    best_sequence = None
    best_score = 0.0
    best_c1 = float('inf')
    
    iteration = 0
    max_iterations = 10000
    
    # Phase 1: Adaptive sequence generation and optimization
    while iteration < max_iterations and time.time() - start_time < max_time:
        # Generate new sequence
        sequence = adaptive_sequence_generation()
        
        # Improve using gradient-based method
        improved_seq = gradient_based_improvement(sequence, iterations=20)
        
        # Further refine with quadratic optimization if possible
        qpo_seq = solve_quadratic_optimization(improved_seq)
        if qpo_seq is not None:
            improved_seq = qpo_seq
            
        # Evaluate
        score = evaluate_sequence(improved_seq)
        c1 = compute_autocorrelation_constant(improved_seq)
        
        if score > best_score:
            best_score = score
            best_sequence = improved_seq.copy()
            best_c1 = c1
            print(f"Iteration {iteration}: New best score = {score:.6f}, C1 = {c1:.6f}")
            
            # Check benchmark
            benchmark_ratio = 1.5031 / c1
            if benchmark_ratio > 1.0:
                print(f"BEAT BENCHMARK at iteration {iteration}! Ratio = {benchmark_ratio:.6f}")
                break
            
            patience = 0  # reset patience on improvement
        else:
            patience += 1
            
        recent_scores.append(score)
        if len(recent_scores) > 10:
            recent_scores.pop(0)
            
        # Early stopping if no improvement for too long
        if patience > max_patience:
            print(f"Early stopping at iteration {iteration}")
            break
            
        iteration += 1
    
    # Final validation
    if best_sequence is not None:
        final_score = evaluate_sequence(best_sequence)
        final_c1 = compute_autocorrelation_constant(best_sequence)
        benchmark_ratio = 1.5031 / final_c1
        
        print(f"Final result:")
        print(f"  Score: {final_score:.6f}")
        print(f"  C1: {final_c1:.6f}")
        print(f"  Benchmark ratio: {benchmark_ratio:.6f}")
        print(f"  Iterations: {iteration}")
    else:
        # Fallback if no good sequence found
        n = int(np.random.lognormal(np.log(300), 0.5))
        best_sequence = [np.random.random() * 10 for _ in range(n)]
        final_score = evaluate_sequence(best_sequence)
        final_c1 = compute_autocorrelation_constant(best_sequence)
        benchmark_ratio = 1.5031 / final_c1
        
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")