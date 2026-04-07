# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import random
import time
from scipy.optimize import minimize

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_c1_constant(sequence):
    """Computes the C1 constant for a given sequence."""
    a = np.array(sequence)
    sum_a = np.sum(a)
    if sum_a < 0.01:
        return float('inf')
    conv = fftconvolve(a, a, mode='full')[:len(a)*2-1]
    max_conv = np.max(conv)
    n = len(a)
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    return c1

def evaluate_sequence(sequence):
    """Evaluates a sequence and returns 1/C1 (the objective to maximize)."""
    try:
        c1 = compute_c1_constant(sequence)
        if c1 == float('inf'):
            return 0.0
        return 1.0 / c1
    except Exception:
        return 0.0

def generate_power_law_sequence(n):
    """Generate a sequence with a power law decay which often performs well."""
    exponents = np.arange(n)
    sequence = np.power(0.95, exponents)
    sequence = np.maximum(sequence, 0.01)  # Minimum value constraint
    return sequence

def generate_step_function(n_steps, heights=None, max_height=1000):
    """Generate a step function with n_steps and optional specific heights."""
    if heights is None:
        # Generate random heights
        heights = [random.uniform(0, max_height) for _ in range(n_steps)]
    else:
        # Ensure we have exactly n_steps heights
        heights = heights[:n_steps] + [0] * (n_steps - len(heights))
        heights = heights[:n_steps]

    # Create step function - evenly distribute heights across the sequence
    total_length = 1000  # Fixed length for step function
    step_sizes = [total_length // n_steps] * n_steps
    # Distribute remaining positions
    remainder = total_length % n_steps
    for i in range(remainder):
        step_sizes[i] += 1

    # Build the sequence
    sequence = []
    for i, (height, size) in enumerate(zip(heights, step_sizes)):
        sequence.extend([height] * size)

    # Trim if necessary
    if len(sequence) > total_length:
        sequence = sequence[:total_length]
    elif len(sequence) < total_length:
        sequence.extend([0.0] * (total_length - len(sequence)))

    return sequence

def compute_convolution_gradient_fft(sequence, epsilon=1e-6):
    """
    Compute the gradient of the maximum convolution value using FFT for efficiency.
    """
    n = len(sequence)
    grad = np.zeros(n)
    base_conv = fftconvolve(sequence, sequence, mode='full')
    max_conv_idx = np.argmax(base_conv[len(sequence)-1:])
    max_conv_val = base_conv[len(sequence)-1:][max_conv_idx]

    # For each element, compute the effect of small perturbation using FFT
    for i in range(n):
        perturbed_seq = sequence.copy()
        perturbed_seq[i] += epsilon
        perturbed_conv = fftconvolve(perturbed_seq, perturbed_seq, mode='full')
        perturbed_max_conv = perturbed_conv[len(sequence)-1:][max_conv_idx]
        grad[i] = (perturbed_max_conv - max_conv_val) / epsilon

    return grad

def gradient_ascent_update(sequence, learning_rate=0.01, max_iterations=10):
    """Perform gradient ascent to improve sequence, directly targeting convolution peak reduction."""
    old_sequence = sequence.copy()

    for iteration in range(max_iterations):
        # Compute gradient of max convolution w.r.t. sequence elements using FFT
        grad = compute_convolution_gradient_fft(sequence)

        # Update using gradient ascent (but avoid increasing max_conv)
        # We want to decrease max_conv while increasing sum(sequence)
        new_sequence = sequence + learning_rate * grad

        # Ensure non-negativity
        new_sequence = np.maximum(new_sequence, 0)

        # Ensure sum is at least 0.01
        if np.sum(new_sequence) < 0.01:
            new_sequence[0] = 0.1

        # Clip to [0, 1000]
        new_sequence = np.clip(new_sequence, 0, 1000)

        # Early stopping if improvement is minimal
        if np.linalg.norm(new_sequence - old_sequence) < 1e-6:
            break

        sequence = new_sequence
        old_sequence = sequence.copy()

    return sequence

def multi_scale_optimization(initial_sequence, max_iter=30):
    """Perform optimization at multiple scales to avoid local optima."""
    sequence = np.array(initial_sequence)
    best_inv_c1 = evaluate_sequence(sequence)
    best_sequence = sequence.copy()

    # Coarse scale: optimize with larger steps
    coarse_lr = 0.1
    for _ in range(3):
        sequence = gradient_ascent_update(sequence, learning_rate=coarse_lr, max_iterations=3)
        inv_c1 = evaluate_sequence(sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = sequence.copy()

    # Medium scale: optimize with medium steps
    medium_lr = 0.05
    for _ in range(5):
        sequence = gradient_ascent_update(sequence, learning_rate=medium_lr, max_iterations=3)
        inv_c1 = evaluate_sequence(sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = sequence.copy()

    # Fine scale: optimize with smaller steps
    fine_lr = 0.01
    for _ in range(10):
        sequence = gradient_ascent_update(sequence, learning_rate=fine_lr, max_iterations=2)
        inv_c1 = evaluate_sequence(sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = sequence.copy()

    # Final adaptive refinement with very small learning rate
    adaptive_lr = 0.005
    for iteration in range(max_iter):
        sequence = gradient_ascent_update(sequence, learning_rate=adaptive_lr, max_iterations=1)
        inv_c1 = evaluate_sequence(sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = sequence.copy()
        # Decrease learning rate over time for stability
        adaptive_lr *= 0.995

    return best_sequence.tolist()

def smart_pattern_initialization(length=None):
    """Create a good initial pattern based on known theoretical structures."""
    if length is None:
        length = random.randint(100, 500)

    # Using a power-law decay which is known to perform well in such problems
    sequence = generate_power_law_sequence(length)

    # Add some randomness to avoid local optima
    noise_factor = 0.05
    sequence = sequence * (1 + np.random.uniform(-noise_factor, noise_factor, length))
    sequence = np.maximum(sequence, 0.01)

    return sequence

def adaptive_crossover(seq1, seq2):
    """Adaptive crossover that preserves high-performing characteristics."""
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    
    padded_seq1 = seq1 + [0] * (max_len - len(seq1))
    padded_seq2 = seq2 + [0] * (max_len - len(seq2))
    
    # Perform crossover with preference towards better performing parents
    new_seq = []
    for i in range(max_len):
        # Probability of inheriting from seq1 increases with fitness
        fitness1 = evaluate_sequence(seq1[:min_len] + [0]*(max_len-min_len)) if len(seq1) <= max_len else 0.0
        fitness2 = evaluate_sequence(seq2[:min_len] + [0]*(max_len-min_len)) if len(seq2) <= max_len else 0.0
        
        prob = fitness1 / (fitness1 + fitness2 + 1e-10) if (fitness1 + fitness2) > 0 else 0.5
        if random.random() < prob:
            new_seq.append(padded_seq1[i])
        else:
            new_seq.append(padded_seq2[i])
    
    return new_seq

def smart_mutate(sequence, mutation_rate=0.1, max_mutation=0.5):
    """Enhanced mutation that considers convolution properties."""
    new_sequence = sequence.copy()
    n = len(sequence)
    
    # Analyze the sequence to identify potentially problematic regions
    conv = fftconvolve(sequence, sequence, mode='full')[:2*n-1]
    max_conv = np.max(conv)
    
    # Mutate with probability adjusted by convolution impact
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            # Adjust mutation magnitude based on sequence characteristics
            mutation_factor = 1.0
            if max_conv > 0 and i < n:  # Only adjust for relevant indices
                # Reduce mutation if this element contributes significantly to convolution max
                # This heuristic promotes stability in high-contribution areas
                contribution_ratio = sequence[i] / max(sequence) if max(sequence) > 0 else 0
                mutation_factor = 1.0 - 0.3 * contribution_ratio  # Less mutation for high-contributing elements
            
            # Apply mutation with adjusted factor
            new_sequence[i] *= random.uniform(
                max(0.5, 1 - max_mutation * mutation_factor),
                min(1.5, 1 + max_mutation * mutation_factor)
            )
            new_sequence[i] = max(0, min(1000, new_sequence[i]))
    
    # Ensure at least one element is positive
    if all(x == 0 for x in new_sequence):
        new_sequence[0] = max(0.1, new_sequence[0])
    
    return new_sequence

def local_improvement_search(initial_sequence, max_iter=100):
    """Improved local search with gradient estimation and simulated annealing."""
    current_sequence = initial_sequence.copy()
    current_fitness = evaluate_sequence(current_sequence)
    best_sequence = current_sequence.copy()
    best_fitness = current_fitness
    
    # Simulated annealing parameters
    temp = 1.0
    cooling_rate = 0.95
    min_temp = 1e-4
    
    for iteration in range(max_iter):
        # Try mutating the current sequence
        mutated = smart_mutate(current_sequence, mutation_rate=0.3, max_mutation=0.2)
        mutated_fitness = evaluate_sequence(mutated)
        
        # Accept or reject based on fitness gain
        if mutated_fitness > current_fitness:
            current_sequence = mutated
            current_fitness = mutated_fitness
        else:
            # Accept with probability based on temperature
            if random.random() < np.exp((mutated_fitness - current_fitness) / (temp + 1e-10)):
                current_sequence = mutated
                current_fitness = mutated_fitness
        
        # Update best
        if current_fitness > best_fitness:
            best_sequence = current_sequence.copy()
            best_fitness = current_fitness
        
        # Cool down temperature
        temp = max(temp * cooling_rate, min_temp)
        
    return best_sequence, best_fitness

def search_for_best_sequence():
    """Main search function using the new hybrid approach."""
    start_time = time.time()
    best_inv_c1 = 0
    best_sequence = None

    # Try multiple initialization strategies
    for attempt in range(20):  # Increased number of attempts for better exploration
        if time.time() - start_time > 170:
            break

        # Initialize with different patterns
        n = random.randint(100, 500)
        init_methods = [
            smart_pattern_initialization(n),
            np.random.uniform(0.01, 100, n),
            np.ones(n) * 10,
            np.array([1.0] * (n // 2) + [0.0] * (n - n // 2)),
            generate_step_function(random.randint(5, 30))
        ]

        for init_method in init_methods:
            sequence = init_method.copy()

            # Multi-scale optimization
            optimized_sequence = multi_scale_optimization(sequence, max_iter=20)

            # Evaluate
            inv_c1 = evaluate_sequence(optimized_sequence)

            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = optimized_sequence[:]

    # If no good sequence found, fallback to a default initialization
    if best_sequence is None:
        best_sequence = smart_pattern_initialization(200)
        best_sequence = multi_scale_optimization(best_sequence)

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")