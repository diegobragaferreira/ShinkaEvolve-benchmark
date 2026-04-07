# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.signal import fftconvolve
from scipy import optimize
import time
import math
from collections import defaultdict

def compute_autocorrelation_constant(sequence):
    """
    Computes the autocorrelation constant C₁ for a given sequence.
    Returns 1/C₁ which we want to maximize.
    """
    if len(sequence) == 0 or np.sum(sequence) < 0.01:
        return 0.0

    # Compute convolution using FFT for efficiency
    conv = fftconvolve(sequence, sequence, mode='full')
    # Take the maximum of the convolution (excluding the zero-padding)
    max_conv = np.max(conv[len(sequence)-1:])

    # Calculate C₁ = 2*n*max(b) / (sum(a))^2
    sum_a = np.sum(sequence)
    n = len(sequence)

    if sum_a == 0:
        return 0.0

    C1 = 2 * n * max_conv / (sum_a ** 2)
    return 1 / C1  # Return reciprocal for maximization

def evaluate_individual(individual):
    """Evaluate an individual (sequence) and return fitness."""
    try:
        fitness = compute_autocorrelation_constant(individual)
        return (fitness,)
    except Exception as e:
        return (0.0,)

def generate_random_sequence(min_length=10, max_length=1000):
    """Generate a random sequence within specified constraints."""
    length = random.randint(min_length, max_length)
    # Generate heights in [0, 1000]
    sequence = [random.uniform(0, 1000) for _ in range(length)]
    return sequence

def get_good_direction_to_move_into(sequence):
    """
    Returns the direction to move into the sequence using finite difference approximation.
    """
    n = len(sequence)
    if n == 0:
        return None

    # Normalize sequence to avoid numerical issues
    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None

    # Use a more principled normalization
    normalized_sequence = np.array(sequence) / sum_sequence

    # Compute current autocorrelation constant
    current_value = compute_autocorrelation_constant(sequence)

    # Approximate gradient using finite differences
    epsilon = 1e-4
    step_direction = np.zeros(n)

    for i in range(n):
        # Create perturbed sequence
        perturbed_sequence = normalized_sequence.copy()
        perturbed_sequence[i] += epsilon

        # Compute new value
        new_value = compute_autocorrelation_constant(perturbed_sequence * sum_sequence)

        # Gradient approximation
        step_direction[i] = (new_value - current_value) / epsilon

    # Normalize the step direction
    step_norm = np.linalg.norm(step_direction)
    if step_norm > 0:
        step_direction = step_direction / step_norm

    # Move in the direction of steepest ascent
    t = 0.01
    new_sequence = (1 - t) * np.array(sequence) + t * step_direction * sum_sequence

    # Ensure non-negativity and clip values
    new_sequence = np.clip(new_sequence, 0, 1000)

    return new_sequence.tolist()

def solve_convolution_lp(f_sequence, rhs):
    """
    Solves the convolution LP for a given sequence and RHS.
    """
    n = len(f_sequence)
    if n == 0:
        return None

    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Build the convolution constraint matrix
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints: b_i >= 0
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

    if result.success:
        g_sequence = result.x
        return g_sequence
    else:
        return None

def initialize_step_function_sequence(length=None):
    """Initialize a step function sequence with optimized characteristics."""
    if length is None:
        length = random.randint(100, 1000)

    # Create a sequence with step-like characteristics
    # We'll alternate between high and low values to reduce convolution peaks
    sequence = []
    for i in range(length):
        if i % 2 == 0:
            # Even indices: higher values
            sequence.append(1000 * np.exp(-i/20))
        else:
            # Odd indices: lower values
            sequence.append(50 * np.exp(-i/100))

    # Normalize to have reasonable total mass
    total_mass = sum(sequence)
    if total_mass > 0:
        sequence = [x / total_mass * 100 for x in sequence]

    return sequence

def evolve_sequences():
    """Evolve sequences to maximize the inverse autocorrelation constant."""
    # Define DEAP structures
    # Note: Instead of DEAP, we implement a simple evolutionary algorithm from scratch
    # This allows us to tune parameters more effectively

    # Set up evolutionary algorithm parameters
    pop_size = 50
    n_generations = 200
    mutation_rate = 0.2
    crossover_rate = 0.5

    # Initialize population
    population = [generate_random_sequence() for _ in range(pop_size)]

    # Evolution loop
    for gen in range(n_generations):
        # Evaluate fitness for all individuals
        fitnesses = [compute_autocorrelation_constant(individual) for individual in population]

        # Selection: tournament selection with adaptive size
        tournament_size = 3 + int(gen / 50)  # Gradually increase selection pressure
        selected_parents = []
        
        for _ in range(pop_size):
            tournament_indices = random.sample(range(pop_size), min(tournament_size, pop_size))
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            selected_parents.append(population[winner_idx][:])  # Copy the selected individual

        # Create new population through crossover and mutation
        new_population = []
        
        # Elitism: keep the best individual
        best_idx = np.argmax(fitnesses)
        new_population.append(population[best_idx][:])

        # Generate offspring
        while len(new_population) < pop_size:
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)

            # Crossover
            if random.random() < crossover_rate and len(parent1) > 1 and len(parent2) > 1:
                crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
                child = parent1[:crossover_point] + parent2[crossover_point:]
            else:
                child = parent1[:]  # Inherit from parent1

            # Mutation
            for i in range(len(child)):
                if random.random() < mutation_rate:
                    # Mutate by adding noise
                    child[i] *= random.uniform(0.8, 1.2)
                    child[i] = max(0, child[i])  # Ensure non-negativity

            new_population.append(child)

        population = new_population[:pop_size]

    # Get best individual
    fitnesses = [compute_autocorrelation_constant(individual) for individual in population]
    best_individual = population[np.argmax(fitnesses)]
    return list(best_individual)

def local_improvement(sequence, max_iter=50):
    """Apply local improvement using gradient-based method."""
    current_seq = sequence[:]
    best_score = compute_autocorrelation_constant(current_seq)
    best_seq = current_seq[:]

    for _ in range(max_iter):
        improved_seq = get_good_direction_to_move_into(current_seq)
        if improved_seq is None:
            break
        new_score = compute_autocorrelation_constant(improved_seq)
        if new_score > best_score:
            best_score = new_score
            best_seq = improved_seq[:]
            current_seq = improved_seq[:]
        else:
            break
    return best_seq

def get_good_direction_to_move_into_lp(sequence):
    """Use LP-based approach for direction finding."""
    n = len(sequence)
    if n == 0:
        return None

    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None

    normalized_sequence = np.array(sequence) / sum_sequence
    rhs = np.max(np.convolve(normalized_sequence, normalized_sequence, mode='full'))
    g_fun = solve_convolution_lp(normalized_sequence, rhs)

    if g_fun is None:
        return None

    sum_g_fun = np.sum(g_fun)
    if sum_g_fun == 0:
        return None

    normalized_g_fun = np.array(g_fun) / sum_g_fun
    t = 0.01
    new_sequence = (1 - t) * np.array(sequence) + t * normalized_g_fun * sum_sequence
    new_sequence = np.clip(new_sequence, 0, 1000)
    return new_sequence.tolist()

def search_for_best_sequence() -> list[float]:
    """Main function to find the best sequence."""
    start_time = time.time()

    # Multi-start approach: try several initialization strategies
    best_score = 0
    best_sequence = None
    strategies = [
        lambda: generate_random_sequence(),
        lambda: initialize_step_function_sequence(),
        lambda: [1.0] * random.randint(100, 1000),
    ]

    for strategy in strategies:
        sequence = strategy()

        # Apply local improvement with gradient and LP methods
        sequence = local_improvement(sequence, max_iter=20)
        lp_improved = get_good_direction_to_move_into_lp(sequence)
        if lp_improved is not None:
            sequence = lp_improved

        # Then evolve using the custom evolutionary algorithm
        try:
            evolved_seq = evolve_sequences()
            evolved_seq = local_improvement(evolved_seq, max_iter=10)
            score = compute_autocorrelation_constant(evolved_seq)
            if score > best_score:
                best_score = score
                best_sequence = evolved_seq[:]
        except Exception as e:
            pass  # Continue to next strategy if evolution fails

    # Fallback to a basic approach if nothing worked
    if best_sequence is None:
        best_sequence = [1.0] * 100

    # Final verification and cleanup
    if len(best_sequence) == 0 or np.sum(best_sequence) < 0.01:
        best_sequence = [1.0]

    # Limit size to prevent excessive computation
    if len(best_sequence) > 1000:
        best_sequence = best_sequence[:1000]

    # Clip values to [0, 1000] for practicality
    best_sequence = [max(0, min(1000, x)) for x in best_sequence]

    elapsed = time.time() - start_time
    # Early exit if time is almost up
    if elapsed > 170:
        return best_sequence

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")