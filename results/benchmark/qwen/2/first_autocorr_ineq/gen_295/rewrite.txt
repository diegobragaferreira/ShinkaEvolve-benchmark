# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
from scipy.signal import fftconvolve
import random
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autocorrelation_constant(sequence):
    """Compute the autocorrelation constant C1 for a sequence."""
    n = len(sequence)
    if n == 0:
        return float('inf')

    # Convert to numpy array for efficient computation
    a = np.array(sequence, dtype=float)

    # Skip if sum is too small to avoid numerical issues
    sum_a = np.sum(a)
    if sum_a < 0.01:
        return float('inf')  # Reject invalid sequences

    # Use FFT for fast convolution for large sequences
    if n > 100:
        padded_length = 2 * n - 1
        a_padded = np.pad(a, (0, padded_length - n), mode='constant')
        a_fft = fft(a_padded)
        autocorr_fft = a_fft * np.conj(a_fft)
        autocorr = ifft(autocorr_fft).real
        autocorr = autocorr[:n]
    else:
        # Use direct convolution for small sequences for precision
        autocorr = np.convolve(a, a, mode='full')[:2*n-1]
        autocorr = autocorr[:n]

    # Find maximum in the autocorrelation
    max_autocorr = np.max(autocorr)

    # Calculate C1 = 2n * max(b) / (sum(a))^2
    c1 = (2 * n * max_autocorr) / (sum_a ** 2)
    return c1

def compute_inverse_c1(sequence):
    """Compute the inverse of C1 (our objective to maximize)."""
    try:
        c1 = compute_autocorrelation_constant(sequence)
        if c1 == float('inf'):
            return 0.0  # Penalty for invalid sequences
        return 1.0 / c1
    except Exception:
        return 0.0

def generate_structured_sequence(length=None, min_length=100, max_length=1000):
    """Generate a structured sequence (step function) with decreasing heights."""
    if length is None:
        length = random.randint(min_length, max_length)
    
    # Create a decreasing step function
    sequence = []
    for i in range(length):
        # Decreasing heights from 1000 down to 1
        height = max(0, 1000 - i * (1000 / length))
        sequence.append(height)
    
    # Ensure at least one element is positive
    if all(x == 0 for x in sequence):
        sequence[0] = 1.0
    
    return sequence

def generate_random_sequence(length=None, min_length=100, max_length=1000):
    """Generate a random sequence with specified or random length."""
    if length is None:
        length = random.randint(min_length, max_length)

    # Generate random heights between 0 and 1000
    sequence = [random.uniform(0, 1000) for _ in range(length)]

    # Ensure at least one element is positive
    if all(x == 0 for x in sequence):
        sequence[0] = 1.0

    return sequence

def mutate_sequence(sequence, mutation_rate=0.1, max_mutation=0.3, generation=0):
    """Apply geometric and random mutations to generate new sequence."""
    new_sequence = sequence.copy()

    # Adapt mutation rate over generations
    adapted_mutation_rate = mutation_rate * (0.9 ** generation)

    # Apply mutations with given probability
    for i in range(len(new_sequence)):
        if random.random() < adapted_mutation_rate:
            # Apply geometric scaling with random factor
            scale_factor = random.uniform(0.5, 2.0)
            new_sequence[i] *= scale_factor

            # Add small noise
            noise_factor = random.uniform(-0.1, 0.1)
            new_sequence[i] = max(0, new_sequence[i] + new_sequence[i] * noise_factor)

            # Clip to valid range [0, 1000]
            new_sequence[i] = max(0, min(1000, new_sequence[i]))

    # Ensure at least one element is positive
    if all(x == 0 for x in new_sequence):
        new_sequence[0] = max(0.1, new_sequence[0])

    return new_sequence

def crossover_sequences(seq1, seq2):
    """Perform crossover between two sequences."""
    # Ensure sequences are of equal length
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))

    # Pad shorter sequence with zeros
    if len(seq1) < max_len:
        seq1.extend([0] * (max_len - len(seq1)))
    if len(seq2) < max_len:
        seq2.extend([0] * (max_len - len(seq2)))

    # Uniform crossover
    child = []
    for i in range(max_len):
        if random.random() < 0.5:
            child.append(seq1[i])
        else:
            child.append(seq2[i])

    return child

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS using scipy.optimize.linprog."""
    try:
        n = len(f_sequence)
        if n < 1:
            return None

        # Create constraint matrix A_ub such that A_ub * x <= b_ub
        # For each convolution index k, we generate constraint coefficients
        num_constraints = 2 * n - 1
        a_ub = np.zeros((num_constraints, n))
        b_ub = np.zeros(num_constraints)

        # Fill constraint matrix
        for k in range(num_constraints):
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    a_ub[k, j] = f_sequence[i]
            b_ub[k] = rhs

        # Add non-negativity constraints (x_i >= 0)
        a_ub_nonneg = -np.eye(n)
        b_ub_nonneg = np.zeros(n)

        # Combine all constraints
        a_ub = np.vstack([a_ub, a_ub_nonneg])
        b_ub = np.hstack([b_ub, b_ub_nonneg])

        # Define objective function (minimize -sum x, i.e., maximize sum x)
        c = -np.ones(n)

        # Solve linear programming problem
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

        if result.success:
            return result.x
        else:
            return None

    except Exception as e:
        return None

def get_good_direction_to_move_into(sequence):
    """Returns a direction to move into the sequence using LP-based optimization."""
    try:
        n = len(sequence)
        if n < 1:
            return None

        # Compute current convolution
        try:
            conv_result = fftconvolve(sequence, sequence, mode='full')[:2*n-1]
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

        # Adaptive step size - decrease with iterations
        base_t = 0.01
        t = base_t * (1.0 / (1.0 + n / 1000.0))  # Decrease with sequence length

        # Solve LP with better initialization and fallback strategies
        g_fun = solve_convolution_lp_with_fallback(normalized_sequence, max_conv)

        if g_fun is None:
            # Try simple gradient ascent as fallback
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

    except Exception as e:
        return None

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

    # Fallback 2: Try with mirrored sequence initialization
    try:
        # Initialize with a mirrored sequence pattern
        mirrored_seq = f_sequence[::-1]
        g_fun = solve_convolution_lp(mirrored_seq, rhs)
        if g_fun is not None:
            return g_fun
    except:
        pass

    # Fallback 3: Return simple pattern
    try:
        # Return a simple uniform pattern
        return np.ones(n) / n
    except:
        pass

    # Fallback 4: Return original sequence (no change)
    return f_sequence

def evolutionary_search(generations=100):
    """Main evolutionary search routine with enhanced strategies."""
    # Initialize population with diverse sequences
    population_size = 30
    population = []

    # Generate a mix of structured and random sequences
    for _ in range(population_size // 2):
        n = random.randint(100, 1000)
        individual = generate_structured_sequence(n)
        population.append(individual)
    
    for _ in range(population_size // 2, population_size):
        n = random.randint(100, 1000)
        individual = generate_random_sequence(n)
        population.append(individual)

    # Evolution parameters
    elite_size = max(1, population_size // 10)  # Top 10% as elite
    mutation_rate = 0.15

    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [(compute_inverse_c1(ind), ind) for ind in population]
        fitness_scores.sort(reverse=True)

        # Keep elite
        elite = [ind for _, ind in fitness_scores[:elite_size]]

        # Create new population
        new_population = elite[:]

        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection with k=3
            parent1 = tournament_selection(population, fitness_scores, k=3)
            parent2 = tournament_selection(population, fitness_scores, k=3)

            offspring = crossover_sequences(parent1, parent2)

            # Apply mutation with adaptive rate
            if random.random() < mutation_rate:
                offspring = mutate_sequence(offspring, mutation_rate=0.15, max_mutation=0.3, generation=gen)

            # Ensure non-negativity
            offspring = [max(0, x) for x in offspring]

            # Ensure minimum sum
            if np.sum(offspring) < 0.01:
                offspring[random.randint(0, len(offspring)-1)] += 0.1

            new_population.append(offspring)

        population = new_population

    # Return best solution
    final_fitness = [(compute_inverse_c1(ind), ind) for ind in population]
    final_fitness.sort(reverse=True)
    return final_fitness[0][1]

def tournament_selection(population, fitness_scores, k=3):
    """Select an individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), k)
    tournament_fitness = [(fitness_scores[i][0], i) for i in tournament_indices]
    winner_idx = max(tournament_fitness)[1]
    return population[winner_idx]

def local_refinement(sequence, iterations=100):
    """Refine a sequence using local search."""
    current_sequence = sequence.copy()
    current_fitness = compute_inverse_c1(current_sequence)

    for _ in range(iterations):
        # Try small modifications to improve fitness
        modified_sequence = mutate_sequence(
            current_sequence, mutation_rate=0.1, max_mutation=0.1, generation=0
        )
        modified_fitness = compute_inverse_c1(modified_sequence)

        # Accept better or equal solutions
        if modified_fitness >= current_fitness:
            current_sequence = modified_sequence
            current_fitness = modified_fitness

    return current_sequence, current_fitness

def gradient_based_improvement(sequence, gradient_iterations=50):
    """Improve sequence using gradient-based approach."""
    current_sequence = np.array(sequence, dtype=float)
    current_fitness = compute_inverse_c1(current_sequence)

    # Simple gradient ascent using finite differences
    step_size = 0.01
    eps = 1e-6

    for iteration in range(gradient_iterations):
        # Approximate gradient using finite differences
        grad = np.zeros_like(current_sequence)
        for i in range(len(current_sequence)):
            # Compute numerical gradient
            delta = np.zeros_like(current_sequence)
            delta[i] = eps
            f_plus = compute_inverse_c1(current_sequence + delta)
            f_minus = compute_inverse_c1(current_sequence - delta)
            grad[i] = (f_plus - f_minus) / (2 * eps)

        # Update step
        new_sequence = current_sequence + step_size * grad

        # Ensure non-negativity and reasonable bounds
        new_sequence = np.maximum(0, new_sequence)
        new_sequence = np.minimum(1000, new_sequence)

        # Check if update improved fitness
        new_fitness = compute_inverse_c1(new_sequence)
        if new_fitness > current_fitness:
            current_sequence = new_sequence
            current_fitness = new_fitness
        else:
            # Reduce step size if no improvement
            step_size *= 0.95
            if step_size < 1e-6:
                break

    return current_sequence.tolist(), current_fitness

def multi_scale_search(max_time_seconds=180):
    """Multi-scale search combining global and local optimization."""
    start_time = time.time()
    best_sequence = None
    best_inv_c1 = 0.0

    # Phase 1: Coarse evolutionary search to find promising regions
    evol_time = max_time_seconds // 3
    evol_seq = evolutionary_search(generations=100)
    evol_fitness = compute_inverse_c1(evol_seq)
    if evol_fitness > best_inv_c1:
        best_inv_c1 = evol_fitness
        best_sequence = evol_seq

    # Phase 2: Fine-grained gradient refinement
    if time.time() - start_time < max_time_seconds:
        grad_seq, grad_fitness = gradient_based_improvement(best_sequence, 100)
        if grad_fitness > best_inv_c1:
            best_inv_c1 = grad_fitness
            best_sequence = grad_seq

    # Phase 3: Local refinement to polish solution
    if time.time() - start_time < max_time_seconds:
        final_seq, final_fitness = local_refinement(best_sequence, 100)
        if final_fitness > best_inv_c1:
            best_inv_c1 = final_fitness
            best_sequence = final_seq

    return best_sequence, best_inv_c1

def search_for_best_sequence():
    """Entry point for search with enhanced optimization strategies."""
    # Try multi-scale search with different starting points
    best_sequence = None
    best_inv_c1 = 0.0

    # Multiple attempts to find the best solution
    for attempt in range(3):
        # Alternate between structured and random initialization
        if attempt % 2 == 0:
            initial_sequence = generate_structured_sequence()
        else:
            initial_sequence = generate_random_sequence()

        # Multi-scale refinement
        refined_seq, refined_fitness = multi_scale_search(60)  # 60 seconds per attempt

        if refined_fitness > best_inv_c1:
            best_inv_c1 = refined_fitness
            best_sequence = refined_seq

    # Final local refinement on the best found solution
    if best_sequence is not None:
        final_seq, final_fitness = local_refinement(best_sequence, 100)
        if final_fitness > best_inv_c1:
            best_inv_c1 = final_fitness
            best_sequence = final_seq

    # If no good solution was found, fall back to a strong baseline
    if best_sequence is None:
        # Start with a structured sequence
        best_sequence = generate_structured_sequence()
        # Local refinement
        best_sequence, best_inv_c1 = local_refinement(best_sequence, 100)

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")