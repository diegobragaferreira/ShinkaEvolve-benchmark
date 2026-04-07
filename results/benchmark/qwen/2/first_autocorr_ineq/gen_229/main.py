# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import random
import time

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def compute_convolution_fft(seq1, seq2):
    """Compute convolution using FFT for efficiency."""
    n = len(seq1)
    # Zero-pad to avoid circular convolution effects
    padded_length = 2 * n - 1
    fft_seq1 = fft(seq1, padded_length)
    fft_seq2 = fft(seq2, padded_length)
    conv_result = ifft(fft_seq1 * np.conj(fft_seq2)).real
    return conv_result[:n]

def compute_c1(sequence):
    """Compute C1 value for a given sequence."""
    n = len(sequence)
    if n < 1:
        return float('inf')

    sum_a = np.sum(sequence)
    if sum_a < 1e-10:
        return float('inf')

    # Compute autoconvolution
    conv_result = compute_convolution_fft(sequence, sequence)
    max_conv = np.max(conv_result)

    # Compute C1
    c1 = 2 * n * max_conv / (sum_a ** 2)
    return c1

def evaluate_sequence(sequence):
    """Evaluate a sequence by computing its inverse C1."""
    c1 = compute_c1(sequence)
    if c1 == float('inf'):
        return 0.0  # Invalid sequence gets low score
    return 1.0 / c1  # Higher inverse C1 is better

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    try:
        n = len(sequence)
        if n < 1:
            return None

        # Normalize the sequence appropriately for LP solving
        sum_sequence = np.sum(sequence)
        if sum_sequence < 1e-10:
            return None

        # Normalize sequence with sqrt(2*n) scaling factor
        normalized_sequence = np.array(sequence) * np.sqrt(2 * n) / sum_sequence

        # Compute convolution using FFT for efficiency
        conv_result = compute_convolution_fft(normalized_sequence, normalized_sequence)
        rhs = np.max(conv_result)

        # Solve the LP problem
        g_fun = solve_convolution_lp(normalized_sequence, rhs)

        if g_fun is None:
            return None

        # Normalize the result again
        sum_g = np.sum(g_fun)
        if sum_g < 1e-10:
            return None

        normalized_g_fun = np.array(g_fun) * np.sqrt(2 * n) / sum_g

        # Use adaptive step size based on sequence complexity and iteration
        # Decrease step size with increasing iteration count and sequence length
        t = 0.05 * np.exp(-0.1 * len(sequence))  # More aggressive decay

        # Create new sequence with adaptive mixing
        new_sequence = (1 - t) * np.array(sequence) + t * normalized_g_fun

        # Ensure non-negativity
        new_sequence = np.maximum(new_sequence, 0)

        return new_sequence.tolist()

    except Exception as e:
        return None

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    try:
        n = len(f_sequence)
        if n < 1:
            return None

        # Sample key convolution indices instead of generating all 2*n-1 constraints
        # This dramatically reduces memory usage and computation time
        max_constraints = min(5000, 2 * n)  # Cap at reasonable number
        constraint_indices = np.linspace(0, 2 * n - 2, min(max_constraints, 2 * n - 1), dtype=int)

        # Sort indices to help with memory access patterns
        constraint_indices = np.sort(constraint_indices)

        # Precompute convolution constraints efficiently
        a_ub = np.zeros((len(constraint_indices), n))
        b_ub = np.zeros(len(constraint_indices))

        for idx, k in enumerate(constraint_indices):
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    a_ub[idx, j] = f_sequence[i]
            b_ub[idx] = rhs

        # Add non-negativity constraints
        a_ub_nonneg = -np.eye(n)
        b_ub_nonneg = np.zeros(n)

        a_ub = np.vstack([a_ub, a_ub_nonneg])
        b_ub = np.hstack([b_ub, b_ub_nonneg])

        # Define objective function (minimize -sum x, i.e., maximize sum x)
        c = -np.ones(n)

        # Solve with error handling
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

        if result.success:
            return result.x
        else:
            # Try alternative solver
            try:
                result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='interior-point')
                if result.success:
                    return result.x
            except:
                pass
            return None

    except Exception as e:
        return None

def create_structured_sequence(n):
    """Create a structured sequence that's more likely to yield good results."""
    # Start with a combination of patterns: exponential decay, uniform, etc.
    # Use a mixture of different approaches to increase diversity

    # Pattern 1: Exponential decay
    exp_decay = np.exp(-np.linspace(0, 3, n))

    # Pattern 2: Uniform distribution
    uniform = np.ones(n)

    # Pattern 3: Random with some structure
    random_part = np.random.rand(n)

    # Combine patterns with weights
    base_seq = 0.5 * exp_decay + 0.3 * uniform + 0.2 * random_part

    # Normalize and scale
    base_seq = base_seq / np.sum(base_seq) * 10

    # Ensure non-negativity and reasonable bounds
    base_seq = np.clip(base_seq, 0, 1000)

    return base_seq.tolist()

def mutate_sequence(sequence, mutation_rate=0.1):
    """Apply random mutation to a sequence."""
    mutated = sequence.copy()
    n = len(mutated)

    # Determine number of mutations based on sequence length and rate
    num_mutations = max(1, int(n * mutation_rate))

    for _ in range(num_mutations):
        idx = random.randint(0, n - 1)
        # Small random change with larger variance
        change_factor = random.uniform(0.5, 1.5)
        mutated[idx] *= change_factor
        mutated[idx] = max(0, mutated[idx])  # Ensure non-negative

    return mutated

def tournament_selection(population, fitnesses, k=5):
    """Select an individual from population using tournament selection."""
    if len(population) < k:
        selected_idx = np.argmax(fitnesses)
        return population[selected_idx]

    tournament_indices = random.sample(range(len(population)), k)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_idx]

def evolutionary_search():
    """Run evolutionary optimization with adaptive parameters."""
    population_size = 30
    generations = 100
    elite_size = 5
    max_time = 170  # Leave 10 seconds for cleanup
    start_time = time.time()
    
    # Initialize population with diverse sequences
    population = []
    for _ in range(population_size):
        # Create sequences of varying lengths
        n = random.randint(100, 1000)
        sequence = create_structured_sequence(n)
        population.append(sequence)

    # Evolution loop
    for gen in range(generations):
        if time.time() - start_time > max_time:
            break
            
        # Evaluate fitness for current population
        fitnesses = [evaluate_sequence(seq) for seq in population]

        # Preserve elite individuals
        elite_indices = np.argsort(fitnesses)[-elite_size:]
        elite_individuals = [population[i] for i in elite_indices]

        # Create new population
        new_population = elite_individuals[:]

        # Fill rest of population through selection, crossover, and mutation
        while len(new_population) < population_size:
            if time.time() - start_time > max_time:
                break
                
            # Tournament selection for parents
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Simple crossover: weighted average of two sequences
            child = []
            min_len = min(len(parent1), len(parent2))
            for i in range(min_len):
                # Blend parents with some randomness
                blend_factor = random.uniform(0.2, 0.8)
                val = blend_factor * parent1[i] + (1 - blend_factor) * parent2[i]
                child.append(val)

            # Extend child if needed
            if len(parent1) > len(parent2):
                child.extend(parent1[len(parent2):])
            elif len(parent2) > len(parent1):
                child.extend(parent2[len(parent1):])
                
            # Mutation with adaptive rate
            mutation_rate = max(0.05, 0.1 - gen * 0.002)  # Decrease over time
            child = mutate_sequence(child, mutation_rate)

            # Ensure minimum sum requirement
            if sum(child) < 0.01:
                child[0] = 0.1

            new_population.append(child)

        population = new_population

    # Return best individual from final population
    final_fitnesses = [evaluate_sequence(seq) for seq in population]
    best_idx = np.argmax(final_fitnesses)
    return population[best_idx]

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence using evolutionary approach."""
    try:
        # Use evolutionary search for better exploration
        best_sequence = evolutionary_search()

        # Final refinement with gradient-based approach
        # Try several iterations of gradient updates
        for _ in range(10):
            refined = get_good_direction_to_move_into(best_sequence)
            if refined is not None:
                best_sequence = refined
            else:
                break

        return best_sequence

    except Exception as e:
        # Fallback to simple approach
        try:
            n = np.random.randint(100, 1000)
            base_seq = create_structured_sequence(n)
            return base_seq
        except:
            return [np.random.random() for _ in range(100)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")