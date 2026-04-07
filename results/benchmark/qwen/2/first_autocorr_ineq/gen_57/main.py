# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy import signal
from scipy.fft import fft, ifft
from scipy.optimize import linprog
import math

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
    
    # Use FFT for fast convolution (more efficient than direct computation)
    # Zero-pad to ensure proper linear convolution
    padded_length = 2 * n - 1
    a_padded = np.pad(a, (0, padded_length - n), mode='constant')
    
    # FFT convolution
    a_fft = fft(a_padded)
    autocorr_fft = a_fft * np.conj(a_fft)
    autocorr = ifft(autocorr_fft).real
    
    # Take only the first n elements for autocorrelation
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

def mutate_sequence(sequence, mutation_rate=0.1, max_mutation=0.3):
    """Apply geometric and random mutations to generate new sequence."""
    new_sequence = sequence.copy()
    
    # Apply mutations with given probability
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
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
        result = linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

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

        # Normalize the sequence
        sum_sequence = np.sum(sequence)
        if sum_sequence < 1e-10:
            return None

        # Normalize sequence
        normalized_sequence = np.array(sequence) / sum_sequence

        # Compute convolution using FFT
        seq_fft = fft(normalized_sequence, 2*n - 1)
        autocorr_fft = seq_fft * np.conj(seq_fft)
        autocorr = ifft(autocorr_fft).real[:n]
        
        # Determine RHS for LP optimization
        rhs = np.max(autocorr)

        # Solve the LP optimization problem to get improvement direction
        g_fun = solve_convolution_lp(normalized_sequence, rhs)

        if g_fun is None:
            return None

        # Normalize the result again
        sum_g = np.sum(g_fun)
        if sum_g < 1e-10:
            return None

        normalized_g_fun = np.array(g_fun) / sum_g

        # Use adaptive step size
        t = 0.01 / (1.0 + 0.001 * n)  # Decrease step size with increasing n

        # Create new sequence with adaptive mixing
        new_sequence = (1 - t) * np.array(sequence) + t * normalized_g_fun * sum_sequence

        # Ensure non-negativity
        new_sequence = np.maximum(new_sequence, 0)

        return new_sequence.tolist()

    except Exception as e:
        return None

def evolutionary_search():
    """Main evolutionary search routine with enhanced strategies."""
    # Initialize population with diverse sequences
    population_size = 30
    population = []
    
    for _ in range(population_size):
        n = random.randint(100, 1000)
        individual = [random.uniform(0, 1000) for _ in range(n)]
        population.append(individual)
    
    # Evolution parameters
    generations = 100
    elite_size = 5
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
            
            # Apply mutation
            if random.random() < mutation_rate:
                offspring = mutate_sequence(offspring, mutation_rate=0.15, max_mutation=0.3)
                
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
            current_sequence, mutation_rate=0.1, max_mutation=0.1
        )
        modified_fitness = compute_inverse_c1(modified_sequence)
        
        # Accept better or equal solutions
        if modified_fitness >= current_fitness:
            current_sequence = modified_sequence
            current_fitness = modified_fitness
    
    return current_sequence, current_fitness

def search_for_best_sequence():
    """Entry point for search with enhanced optimization strategies."""
    # Try multiple random starting points
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Try several strategies to find the best solution
    for attempt in range(5):
        # Generate initial sequence with different strategies
        if attempt == 0:
            # Use a simple random starting point
            initial_sequence = generate_random_sequence()
        else:
            # Use a more structured approach or evolutionary search
            initial_sequence = evolutionary_search()
            
        # Try local refinement
        refined_seq, refined_fitness = local_refinement(initial_sequence, 50)
        
        if refined_fitness > best_inv_c1:
            best_inv_c1 = refined_fitness
            best_sequence = refined_seq
            
        # Try LP-based improvement
        lp_improved = get_good_direction_to_move_into(initial_sequence)
        if lp_improved is not None:
            lp_fitness = compute_inverse_c1(lp_improved)
            if lp_fitness > best_inv_c1:
                best_inv_c1 = lp_fitness
                best_sequence = lp_improved

    # Final local refinement on the best found solution
    if best_sequence is not None:
        final_seq, final_fitness = local_refinement(best_sequence, 100)
        if final_fitness > best_inv_c1:
            best_inv_c1 = final_fitness
            best_sequence = final_seq

    return best_sequence if best_sequence is not None else generate_random_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
