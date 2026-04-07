# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import random
import time

def compute_convolution_fft(seq):
    """Compute convolution using FFT for better performance."""
    n = len(seq)
    if n < 1:
        return np.array([])
    padded_seq = np.pad(seq, (0, n-1), mode='constant')
    fft_seq = fft(padded_seq)
    conv_result = ifft(fft_seq * np.conj(fft_seq)).real
    return conv_result[:2*n-1]

def calculate_fitness(sequence):
    """Calculate fitness as inverse of C1, i.e., (sum(a))^2 / (2*n*max(conv))"""
    n = len(sequence)
    if n < 1:
        return 0.0

    sum_a = np.sum(sequence)
    if sum_a < 1e-10:
        return 0.0

    conv = compute_convolution_fft(sequence)
    max_conv = np.max(conv)

    if max_conv < 1e-10:
        return 0.0

    # Calculate fitness: (sum(a))^2 / (2*n*max(conv))
    fitness = (sum_a ** 2) / (2 * n * max_conv)
    return fitness

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence with improved strategies."""
    n = len(sequence)
    if n < 1:
        return None

    # Compute current convolution
    try:
        conv_result = compute_convolution_fft(sequence)
        max_conv = np.max(conv_result)
    except Exception:
        # Fallback to direct convolution if FFT fails
        conv_result = np.convolve(sequence, sequence)
        max_conv = np.max(conv_result)

    # Normalize sequence for better numerics
    sum_sequence = np.sum(sequence)
    if sum_sequence < 1e-10:
        return None

    normalized_sequence = [x / sum_sequence for x in sequence]

    # Adaptive step size
    base_t = 0.01
    t = base_t * (1.0 / (1.0 + n / 1000.0))

    # Attempt to solve LP for a better update direction
    g_fun = solve_convolution_lp_with_fallback(normalized_sequence, max_conv)

    if g_fun is None:
        # Fallback to simple gradient ascent
        try:
            # Simple gradient ascent - move towards increasing values
            g_fun = [max(0, x + 0.01 * (random.random() - 0.5)) for x in sequence]
            # Normalize again
            sum_g = np.sum(g_fun)
            if sum_g > 0:
                g_fun = [x / sum_g for x in g_fun]
        except:
            return None

    # Apply the update
    if g_fun is not None:
        sum_g = np.sum(g_fun)
        if sum_g > 0:
            normalized_g_fun = [x / sum_g for x in g_fun]
            new_sequence = [
                (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
            ]
            return new_sequence

    return sequence

def solve_convolution_lp_with_fallback(f_sequence, rhs):
    """Solves the convolution LP with fallback strategies."""
    n = len(f_sequence)
    if n < 1:
        return None

    # Try normal LP approach first
    g_fun = solve_convolution_lp(f_sequence, rhs)

    if g_fun is not None:
        return g_fun

    # Fallback 1: Try with slightly relaxed constraints
    try:
        # Relax the constraint slightly
        g_fun = solve_convolution_lp(f_sequence, rhs * 1.01)
        if g_fun is not None:
            return g_fun
    except:
        pass

    # Fallback 2: Return symmetric pattern (mirrored)
    try:
        # Return a symmetric pattern
        g_fun = [f_sequence[n//2]] * n
        sum_g = np.sum(g_fun)
        if sum_g > 0:
            g_fun = [x / sum_g for x in g_fun]
            return g_fun
    except:
        pass

    # Fallback 3: Return simple uniform pattern
    try:
        return np.ones(n) / n
    except:
        pass

    # Fallback 4: Return original sequence (no change)
    return f_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    if n < 1:
        return None

    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Generate constraint matrix using FFT for efficiency
    try:
        for k in range(2 * n - 1):
            row = np.zeros(n)
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    row[j] = f_sequence[i]
            a_ub.append(row)
            b_ub.append(rhs)
    except:
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

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            return None
    except:
        # If optimization fails, return None to trigger fallback
        return None

def adaptive_gradient_update(sequence, iteration, max_iterations):
    """Perform adaptive gradient update with decreasing step size."""
    n = len(sequence)
    if n < 1:
        return sequence
    
    # Adaptive step size: decreases with iteration
    base_step = 0.05
    step_size = base_step * (1.0 - iteration / max_iterations)
    
    # Compute current convolution and gradients
    conv = compute_convolution_fft(sequence)
    sum_a = np.sum(sequence)
    
    if sum_a < 1e-10 or np.max(conv) < 1e-10:
        return sequence
    
    # Simple gradient estimation: increase smaller values, decrease larger ones
    grad = np.array(sequence) - np.mean(sequence)
    grad = np.clip(grad, -0.1, 0.1)  # Clip to prevent extreme moves
    
    # Apply gradient update
    new_sequence = np.array(sequence) + step_size * grad
    
    # Ensure non-negativity and normalize
    new_sequence = np.maximum(new_sequence, 0)
    sum_new = np.sum(new_sequence)
    
    if sum_new > 0:
        new_sequence = new_sequence / sum_new
    
    return new_sequence.tolist()

def adaptive_sequence_length_adjustment(current_len, fitness_history, patience=5):
    """Adjust sequence length based on recent fitness improvements."""
    if len(fitness_history) < patience + 1:
        return current_len
    
    recent_improvements = [
        fitness_history[-i] - fitness_history[-i-1] 
        for i in range(1, min(patience, len(fitness_history)-1))
    ]
    
    avg_improvement = np.mean(recent_improvements) if recent_improvements else 0
    
    # Increase length if recent improvements are positive
    if avg_improvement > 0.001:
        new_len = min(current_len + 10, 2000)  # Cap at 2000
    elif avg_improvement < -0.001 and current_len > 100:
        new_len = max(current_len - 10, 100)  # Don't go below 100
    else:
        new_len = current_len  # No change
    
    return new_len

def restart_strategy(sequence, fitness_history, max_fitness):
    """Restart with a new random sequence if no improvement after several iterations."""
    if len(fitness_history) < 10:
        return False, sequence
    
    recent_fitness = fitness_history[-10:]
    if max(recent_fitness) <= max_fitness * 0.99:
        return True, [random.random() * 10 for _ in range(len(sequence))]
    return False, sequence

def search_for_best_sequence() -> list[float]:
    """Main function implementing adaptive gradient evolution."""
    # Initialize parameters
    start_time = time.time()
    max_time = 170  # Leave some buffer for cleanup
    max_iter = 1000
    patience = 10
    current_sequence = [random.random() * 10 for _ in range(100)]
    fitness_history = []
    max_fitness = 0.0
    best_sequence = current_sequence[:]
    
    # Multi-start strategy to avoid local minima
    num_starts = 10
    for start_idx in range(num_starts):
        if time.time() - start_time > max_time:
            break

        # Randomly initialize sequence length and values
        n = np.random.randint(100, 1000)
        current_sequence = [random.random() * 10 for _ in range(n)]

        for iteration in range(max_iter):
            if time.time() - start_time > max_time:
                break

            # Update sequence length adaptively
            if iteration % 50 == 0 and iteration > 0:
                current_len = len(current_sequence)
                new_len = adaptive_sequence_length_adjustment(current_len, fitness_history)
                if new_len != current_len:
                    # Adjust sequence length
                    if new_len > current_len:
                        current_sequence.extend([0.0] * (new_len - current_len))
                    else:
                        current_sequence = current_sequence[:new_len]
            
            # Perform adaptive gradient update
            new_sequence = adaptive_gradient_update(current_sequence, iteration, max_iter)
            
            # Evaluate new fitness
            new_fitness = calculate_fitness(new_sequence)
            fitness_history.append(new_fitness)
            
            # Check for improvement
            if new_fitness > max_fitness:
                max_fitness = new_fitness
                best_sequence = new_sequence[:]
            
            # Apply restart strategy
            should_restart, new_seq = restart_strategy(current_sequence, fitness_history, max_fitness)
            if should_restart:
                current_sequence = new_seq
            else:
                current_sequence = new_sequence

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
