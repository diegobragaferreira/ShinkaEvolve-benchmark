# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize, signal
from scipy.fft import fft, ifft
import random
import time
from typing import List, Tuple

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_c1_constant(sequence: List[float]) -> float:
    """
    Computes the C1 constant for a given sequence.
    C1 = 2n * max(convolution) / (sum(sequence))^2
    Returns 1/C1 (objective to maximize).
    """
    a = np.array(sequence)
    sum_a = np.sum(a)
    if sum_a < 0.01:
        return 0.0  # Invalid sequence
    
    # Compute autoconvolution using FFT for efficiency
    conv = signal.fftconvolve(a, a, mode='full')[:len(a)*2-1]
    max_conv = np.max(conv)
    n = len(a)
    
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    if c1 == 0:
        return 0.0
    return 1.0 / c1

def get_good_direction_to_move_into(
    sequence: list[float],
    iteration: int = 0
) -> list[float] | None:
    """Returns the direction to move into the sequence with adaptive learning rate."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    if sum_sequence < 1e-10:
        return None
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    rhs = np.max(signal.fftconvolve(normalized_sequence, normalized_sequence, mode='full')[:2*n-1])
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None
    sum_sequence = np.sum(g_fun)
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_sequence for x in g_fun]
    # Adaptive learning rate with exponential decay
    base_t = 0.01
    t = base_t * np.exp(-iteration / 100.0)
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
            return None
    except:
        return None

def spectral_analysis(sequence: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform spectral analysis of a sequence to understand its frequency characteristics.
    Returns dominant frequencies and power distribution.
    """
    n = len(sequence)
    if n < 1:
        return np.array([]), np.array([])
    
    # Compute discrete Fourier transform
    fft_coeffs = fft(sequence)
    # Power spectrum (magnitude squared)
    power_spectrum = np.abs(fft_coeffs)**2
    
    # Return frequency indices and corresponding powers
    freq_indices = np.arange(n)
    return freq_indices, power_spectrum

def create_spectral_guided_sequence(n: int) -> List[float]:
    """
    Create a sequence informed by spectral analysis to encourage good convolution properties.
    """
    # Create an initial sequence that tends to produce favorable autoconvolutions
    # Using a combination of exponential decay and sinusoidal patterns
    t = np.linspace(0, 4*np.pi, n)
    
    # Exponential decay to reduce high-frequency content
    exp_decay = np.exp(-t/2)
    
    # Sinusoidal modulation to add some regularity
    sine_mod = 0.5 + 0.5 * np.sin(t)
    
    # Random component for diversity
    random_part = np.random.rand(n)
    
    # Combine with different weights to balance structure and randomness
    base_seq = 0.4 * exp_decay + 0.3 * sine_mod + 0.3 * random_part
    
    # Normalize to unit sum
    base_seq = base_seq / np.sum(base_seq) * 10
    
    # Ensure bounds
    base_seq = np.clip(base_seq, 0, 1000)
    
    return base_seq.tolist()

def spectral_filter(sequence: List[float], alpha: float = 0.5) -> List[float]:
    """
    Apply spectral filtering to smooth the sequence and emphasize preferred frequency components.
    """
    try:
        n = len(sequence)
        if n < 1:
            return sequence
            
        # Get spectral info
        _, power_spectrum = spectral_analysis(sequence)
        
        # Identify dominant frequencies (top 40% by power)
        threshold = np.percentile(power_spectrum, 60)
        dominant_freqs = np.where(power_spectrum > threshold)[0]
        
        # Create frequency domain filter
        filtered_spectrum = np.zeros(n)
        
        # Emphasize dominant frequencies
        filtered_spectrum[dominant_freqs] = 1.0
        
        # Apply smoothing to the filter
        filtered_spectrum = signal.savgol_filter(filtered_spectrum, 5, 1)
        
        # Ensure non-negative
        filtered_spectrum = np.maximum(filtered_spectrum, 0)
        
        # Apply filter in frequency domain
        seq_fft = fft(sequence)
        filtered_fft = seq_fft * filtered_spectrum
        filtered_seq = np.real(ifft(filtered_fft))
        
        # Ensure non-negativity and normalize
        filtered_seq = np.maximum(filtered_seq, 0)
        filtered_seq = filtered_seq / np.sum(filtered_seq) * 10
        filtered_seq = np.clip(filtered_seq, 0, 1000)
        
        return filtered_seq.tolist()
        
    except Exception:
        return sequence

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """
    Mutate a sequence with controlled variation.
    """
    mutated = sequence.copy()
    n = len(mutated)
    
    # Determine number of mutations
    num_mutations = max(1, int(n * mutation_rate))
    
    for _ in range(num_mutations):
        idx = random.randint(0, n - 1)
        # Multiplicative mutation
        delta = random.uniform(0.8, 1.2)
        mutated[idx] *= delta
        mutated[idx] = max(0, mutated[idx])
    
    return mutated

def crossover_sequences(parent1: List[float], parent2: List[float]) -> List[float]:
    """
    Perform crossover between two sequences.
    """
    min_len = min(len(parent1), len(parent2))
    
    # Point crossover
    crossover_point = random.randint(1, min_len - 1)
    
    child = parent1[:crossover_point] + parent2[crossover_point:]
    
    # Extend if needed
    if len(parent1) > min_len:
        child.extend(parent1[min_len:])
    elif len(parent2) > min_len:
        child.extend(parent2[min_len:])
        
    return child

def tournament_selection(population: List[List[float]], 
                         fitnesses: List[float], 
                         k: int = 3) -> List[float]:
    """
    Select an individual from population using tournament selection.
    """
    if len(population) < k:
        selected_idx = np.argmax(fitnesses)
        return population[selected_idx]

    tournament_indices = random.sample(range(len(population)), k)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_idx]

def evolutionary_search(max_time_seconds: int = 180) -> List[float]:
    """
    Run evolutionary optimization guided by spectral properties.
    """
    start_time = time.time()
    
    # Parameters
    population_size = 30
    generations = 50
    elite_size = 6
    mutation_rate = 0.1
    
    # Initialize diverse population
    population = []
    for _ in range(population_size):
        # Varying sequence lengths
        n = random.randint(80, 1000)
        sequence = create_spectral_guided_sequence(n)
        population.append(sequence)
    
    # Evolution loop
    for gen in range(generations):
        if time.time() - start_time > max_time_seconds - 2:
            break
            
        # Evaluate fitness
        fitnesses = [compute_c1_constant(seq) for seq in population]
        
        # Preserve elite
        elite_indices = np.argsort(fitnesses)[-elite_size:]
        elite_individuals = [population[i] for i in elite_indices]
        
        # Create new population
        new_population = elite_individuals[:]
        
        # Fill rest with offspring
        while len(new_population) < population_size:
            # Parent selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Crossover
            child = crossover_sequences(parent1, parent2)
            
            # Apply spectral filtering to child
            filtered_child = spectral_filter(child)
            
            # Mutation
            mutated_child = mutate_sequence(filtered_child, mutation_rate)
            
            # Ensure minimum sum
            if sum(mutated_child) < 0.01:
                mutated_child[0] = 0.1
                
            new_population.append(mutated_child)
            
        population = new_population
    
    # Return best individual
    final_fitnesses = [compute_c1_constant(seq) for seq in population]
    best_idx = np.argmax(final_fitnesses)
    return population[best_idx]

def local_refinement(sequence: List[float], max_iter: int = 50) -> List[float]:
    """
    Perform local refinement to improve the solution.
    """
    current = sequence.copy()
    current_fitness = compute_c1_constant(current)
    
    for iteration in range(max_iter):
        if time.time() - start_time > 180 - 2:
            break
            
        # Try various perturbations
        candidates = []
        
        # Small random perturbations
        for _ in range(5):
            perturbed = current.copy()
            idx = random.randint(0, len(perturbed) - 1)
            delta = random.uniform(-0.05, 0.05) * perturbed[idx] if perturbed[idx] > 0 else random.uniform(-10, 10)
            perturbed[idx] = max(0, min(1000, perturbed[idx] + delta))
            candidates.append(perturbed)
        
        # Spectral guided perturbation
        try:
            spectral_perturbed = spectral_filter(current)
            candidates.append(spectral_perturbed)
        except:
            pass
        
        # Gradient-based move with LP optimization and adaptive learning rate
        try:
            gradient_move = get_good_direction_to_move_into(current, iteration)
            if gradient_move is not None:
                candidates.append(gradient_move)
        except:
            pass
        
        # Evaluate and accept best
        for candidate in candidates:
            fitness = compute_c1_constant(candidate)
            if fitness > current_fitness:
                current = candidate
                current_fitness = fitness
                
    return current

def adaptive_gradient_update(sequence, iteration, max_iterations):
    """Perform adaptive gradient update with decreasing step size."""
    n = len(sequence)
    if n < 1:
        return sequence
    
    # Adaptive step size: decreases with iteration
    base_step = 0.05
    step_size = base_step * (1.0 - iteration / max_iterations)
    
    # Compute current convolution and gradients
    conv = signal.fftconvolve(sequence, sequence, mode='full')[:len(sequence)*2-1]
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

def search_for_best_sequence() -> List[float]:
    """
    Main function to search for the best coefficient sequence.
    """
    global start_time
    start_time = time.time()
    
    # Run evolutionary search
    best_sequence = evolutionary_search(170)
    
    # Local refinement
    refined_sequence = local_refinement(best_sequence, 100)
    
    # Final check
    final_fitness = compute_c1_constant(refined_sequence)
    if final_fitness <= 0.0:
        # If refinement failed, fallback to simple sequence
        n = random.randint(100, 1000)
        return [random.uniform(0.1, 1.0) for _ in range(n)]
    
    return refined_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
