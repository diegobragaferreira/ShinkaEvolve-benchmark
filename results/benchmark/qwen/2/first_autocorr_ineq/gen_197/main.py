# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import random
import time

def convolve_fft(a, b):
    """Compute convolution using FFT for better performance."""
    n = len(a)
    # Zero-pad to avoid circular convolution effects
    padded_length = 2 * n - 1
    fa = fft(a, padded_length)
    fb = fft(b, padded_length)
    result = ifft(fa * fb).real
    return result[:n]

def get_good_direction_to_move_into(
    sequence: list[float],
    iteration: int = 0,
) -> list[float] | None:
    """Returns the direction to move into the sequence with adaptive step size and momentum."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    if sum_sequence < 1e-10:
        return None
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    rhs = np.max(convolve_fft(normalized_sequence, normalized_sequence))
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None
    sum_sequence = np.sum(g_fun)
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_sequence for x in g_fun]

    # Adaptive step size with exponential decay and momentum
    base_t = 0.02
    t = base_t * np.exp(-iteration / 150.0)  # Slower decay for more stable convergence

    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    return new_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints: b_i >= 0
    a_ub_nonneg = -np.eye(n)  # Negative identity matrix for b_i >= 0
    b_ub_nonneg = np.zeros(n)  # Zero vector

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            # LP optimization failed, try advanced fallback strategies
            # Fallback 1: Mirrored symmetric pattern
            try:
                center_idx = n // 2
                g_fun = [f_sequence[center_idx]] * n
                sum_g = np.sum(g_fun)
                if sum_g > 0:
                    g_fun = [x / sum_g for x in g_fun]
                    return g_fun
            except:
                pass

            # Fallback 2: Gaussian-like pattern
            try:
                g_fun = [np.exp(-((i - n//2)**2) / (2 * (n//10)**2)) for i in range(n)]
                sum_g = np.sum(g_fun)
                if sum_g > 0:
                    g_fun = [x / sum_g for x in g_fun]
                    return g_fun
            except:
                pass

            # Fallback 3: Uniform distribution
            try:
                return np.ones(n) / n
            except:
                pass

            # Fallback 4: Random pattern with normalization
            try:
                g_fun = [random.random() for _ in range(n)]
                sum_g = np.sum(g_fun)
                if sum_g > 0:
                    g_fun = [x / sum_g for x in g_fun]
                    return g_fun
            except:
                pass

            # Fallback 5: Return original sequence (no change)
            return f_sequence
    except:
        # Even fallbacks failed, return original sequence
        return f_sequence

def calculate_fitness(sequence):
    """Calculate fitness as inverse of C1, i.e., (sum(a))^2 / (2*n*max(conv))"""
    n = len(sequence)
    if n < 1:
        return 0.0

    sum_a = np.sum(sequence)
    if sum_a < 1e-10:
        return 0.0

    conv = convolve_fft(sequence, sequence)
    max_conv = np.max(conv)

    if max_conv < 1e-10:
        return 0.0

    # Calculate fitness: (sum(a))^2 / (2*n*max(conv))
    fitness = (sum_a ** 2) / (2 * n * max_conv)
    return fitness

def adaptive_sequence_length_adjustment(current_len, fitness_history, patience=5):
    """Adjust sequence length based on recent fitness improvements."""
    if len(fitness_history) < patience + 1:
        return current_len

    recent_improvements = [
        fitness_history[-i] - fitness_history[-i-1]
        for i in range(1, min(patience, len(fitness_history)-1))
    ]

    avg_improvement = np.mean(recent_improvements) if recent_improvements else 0

    # More aggressive adjustments
    if avg_improvement > 0.002:
        new_len = min(current_len + 30, 2000)  # Cap at 2000
    elif avg_improvement < -0.002 and current_len > 100:
        new_len = max(current_len - 30, 100)  # Don't go below 100
    else:
        new_len = current_len  # No change

    return new_len

def generate_initializations(n):
    """Generate diverse initializations for better exploration."""
    initializations = []

    # Strategy 1: Uniform distribution
    initializations.append(np.ones(n) / n)

    # Strategy 2: Exponential decay
    exp_decay = [np.exp(-i/100.0) for i in range(n)]
    exp_decay = np.array(exp_decay)
    exp_decay = exp_decay / np.sum(exp_decay)
    initializations.append(exp_decay)

    # Strategy 3: Random with normalization
    random_init = [random.random() for _ in range(n)]
    random_init = np.array(random_init)
    random_init = random_init / np.sum(random_init)
    initializations.append(random_init)

    # Strategy 4: Gaussian-like shape
    gaussian_like = [np.exp(-((i - n//2)**2) / (2 * (n//10)**2)) for i in range(n)]
    gaussian_like = np.array(gaussian_like)
    gaussian_like = gaussian_like / np.sum(gaussian_like)
    initializations.append(gaussian_like)

    # Strategy 5: Peaks at edges
    edge_peaks = np.zeros(n)
    edge_peaks[0] = 0.5
    edge_peaks[-1] = 0.5
    edge_peaks = edge_peaks / np.sum(edge_peaks)
    initializations.append(edge_peaks)

    return initializations

def ensemble_direction_selection(current_sequence, iteration):
    """Select direction using ensemble of different strategies."""
    n = len(current_sequence)

    # Get multiple directions from different strategies
    directions = []

    # Strategy 1: Normal optimization step
    normal_direction = get_good_direction_to_move_into(current_sequence, iteration)
    if normal_direction is not None:
        directions.append(normal_direction)

    # Strategy 2: Gradient-based (simple)
    grad_direction = simple_gradient_update(current_sequence)
    if grad_direction is not None:
        directions.append(grad_direction)

    # Strategy 3: Perturbation-based
    pert_direction = perturbation_based_update(current_sequence)
    if pert_direction is not None:
        directions.append(pert_direction)

    # Strategy 4: Ensemble averaging
    if len(directions) > 1:
        avg_direction = np.mean([np.array(d) for d in directions], axis=0)
        avg_direction = np.maximum(avg_direction, 0)
        sum_avg = np.sum(avg_direction)
        if sum_avg > 0:
            avg_direction = avg_direction / sum_avg
            directions.append(avg_direction.tolist())

    # If we have at least one valid direction, pick the best one
    if directions:
        fitnesses = [calculate_fitness(d) for d in directions]
        best_idx = np.argmax(fitnesses)
        return directions[best_idx]

    # If no valid directions, return current sequence
    return current_sequence

def simple_gradient_update(sequence):
    """Simple gradient-based update."""
    n = len(sequence)
    if n < 1:
        return None

    # Simple gradient: move toward larger values
    grad = np.array(sequence) - np.mean(sequence)
    grad = np.clip(grad, -0.05, 0.05)

    new_sequence = np.array(sequence) + 0.01 * grad
    new_sequence = np.maximum(new_sequence, 0)

    sum_new = np.sum(new_sequence)
    if sum_new > 0:
        new_sequence = new_sequence / sum_new
        return new_sequence.tolist()
    return None

def perturbation_based_update(sequence):
    """Update by adding small random perturbations."""
    n = len(sequence)
    if n < 1:
        return None

    # Add small random perturbations
    perturbed = [max(0, x + 0.01 * (random.random() - 0.5)) for x in sequence]

    sum_pert = np.sum(perturbed)
    if sum_pert > 0:
        perturbed = [x / sum_pert for x in perturbed]
        return perturbed

    return None

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence with enhanced strategies."""
    start_time = time.time()
    max_time = 170  # Leave some buffer for cleanup
    best_sequence = None
    best_fitness = 0.0
    elite_sequences = []  # Store top performing sequences
    fitness_history = []  # Track fitness over iterations

    # Multi-start strategy to avoid local minima
    num_starts = 20  # Increased number for better exploration
    for start_idx in range(num_starts):
        if time.time() - start_time > max_time:
            break

        # Try different initialization strategies
        n = np.random.randint(100, 1000)
        initializations = generate_initializations(n)
        best_init_fitness = 0.0
        best_init_sequence = None

        for init in initializations:
            current_sequence = init.tolist()
            current_fitness = calculate_fitness(current_sequence)
            if current_fitness > best_init_fitness:
                best_init_fitness = current_fitness
                best_init_sequence = current_sequence[:]

        if best_init_sequence is None:
            # Fallback to simple random
            current_sequence = [random.random() * 10 for _ in range(n)]
        else:
            current_sequence = best_init_sequence

        # Local search with multiple iterations
        local_max_iter = 100
        for iter_idx in range(local_max_iter):
            if time.time() - start_time > max_time:
                break

            # Adjust sequence length adaptively every 15 iterations
            if iter_idx % 15 == 0 and iter_idx > 0:
                n = adaptive_sequence_length_adjustment(n, fitness_history)
                if n != len(current_sequence):
                    # Adjust sequence length
                    if n > len(current_sequence):
                        current_sequence.extend([0.0] * (n - len(current_sequence)))
                    else:
                        current_sequence = current_sequence[:n]

            # Use ensemble selection for next direction
            h_function = ensemble_direction_selection(current_sequence, iter_idx)
            if h_function is not None:
                current_sequence = h_function
            else:
                # If we can't improve, try a new random sequence
                n = np.random.randint(100, 1000)
                current_sequence = [random.random() * 10 for _ in range(n)]

            # Evaluate fitness
            current_fitness = calculate_fitness(current_sequence)
            fitness_history.append(current_fitness)

            # Store elite sequences
            if len(elite_sequences) < 10 or current_fitness > elite_sequences[0][0]:
                elite_sequences.append((current_fitness, current_sequence[:]))
                elite_sequences.sort(key=lambda x: x[0], reverse=True)
                elite_sequences = elite_sequences[:10]  # Keep only top 10

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_sequence = current_sequence[:]

    # Final check to ensure we have a valid sequence
    if best_sequence is None:
        n = np.random.randint(100, 1000)
        best_sequence = [random.random() * 10 for _ in range(n)]

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")