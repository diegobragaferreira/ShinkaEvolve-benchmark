# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.signal import fftconvolve
from scipy import optimize
import time

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

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

def generate_exponential_sequence(length):
    """Generate an exponentially decaying sequence."""
    sequence = [1000 * np.exp(-i/10) for i in range(length)]
    total_mass = sum(sequence)
    if total_mass > 0:
        sequence = [x / total_mass * 100 for x in sequence]
    return sequence

def generate_periodic_sequence(length):
    """Generate a periodic-like sequence."""
    sequence = []
    period = max(10, length // 20)
    for i in range(length):
        sequence.append(1000 * np.sin(i / period) + 100)
    total_mass = sum(sequence)
    if total_mass > 0:
        sequence = [x / total_mass * 100 for x in sequence]
    return sequence

def generate_spread_sequence(length):
    """Generate a spread-out sequence with sparse peaks."""
    sequence = [0.0] * length
    num_peaks = max(2, min(10, length // 50))
    peak_positions = sorted(random.sample(range(length), num_peaks))
    for pos in peak_positions:
        sequence[pos] = 1000
    total_mass = sum(sequence)
    if total_mass > 0:
        sequence = [x / total_mass * 100 for x in sequence]
    return sequence

def generate_random_sequence(min_length=10, max_length=1000):
    """Generate a random sequence within specified constraints."""
    length = random.randint(min_length, max_length)
    # Generate heights in [0, 1000]
    sequence = [random.uniform(0, 1000) for _ in range(length)]
    return sequence

def generate_sequence_by_pattern(length):
    """Generates a sequence based on pattern preference."""
    patterns = [
        generate_exponential_sequence,
        generate_periodic_sequence,
        generate_spread_sequence,
    ]
    return random.choice(patterns)(length)

def get_pattern_score(sequence):
    """Assign a score to the sequence based on its structure."""
    # Measure how evenly distributed the sequence is (lower is better for avoiding peaks)
    std_dev = np.std(sequence)
    # Measure concentration (higher is better if the sequence is concentrated)
    mean_val = np.mean(sequence)
    if mean_val > 0:
        concentration = np.sum([(x-mean_val)**2 for x in sequence]) / len(sequence) / mean_val
    else:
        concentration = 0
    return std_dev, concentration

def adaptive_initialize_sequence(length=None):
    """Initialize a sequence using adaptive pattern selection."""
    if length is None:
        length = random.randint(100, 1000)

    # Strategy selection based on empirical knowledge
    strategy = random.choices(
        ['exp', 'periodic', 'spread', 'random'],
        weights=[0.35, 0.25, 0.25, 0.15]  # Prefer exponential and periodic
    )[0]

    if strategy == 'exp':
        return generate_exponential_sequence(length)
    elif strategy == 'periodic':
        return generate_periodic_sequence(length)
    elif strategy == 'spread':
        return generate_spread_sequence(length)
    else:
        return generate_random_sequence(length, length)

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

    # Approximate gradient using finite differences with adaptive epsilon
    epsilon = 1e-5
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

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
    except Exception:
        try:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='interior-point')
        except Exception:
            return None

    if result.success:
        g_sequence = result.x
        return g_sequence
    else:
        return None

def adaptive_evolve_sequences():
    """Evolve sequences with adaptive strategies to maximize 1/C₁."""
    pop_size = 60
    n_generations = 200
    best_score = 0
    best_sequence = []

    # Initialize population with adaptive strategies
    population = [adaptive_initialize_sequence(random.randint(100, 1000)) for _ in range(pop_size)]

    for gen in range(n_generations):
        # Evaluate fitness
        fitnesses = [compute_autocorrelation_constant(individual) for individual in population]
        
        # Track best solution
        current_best_idx = np.argmax(fitnesses)
        current_best_score = fitnesses[current_best_idx]
        
        if current_best_score > best_score:
            best_score = current_best_score
            best_sequence = population[current_best_idx][:]

        # Adaptive tournament selection based on fitness diversity
        fitness_std = np.std(fitnesses)
        if fitness_std < 0.01 and gen > 100:
            # If diversity is low, increase selection pressure
            tournament_size = 8
        else:
            tournament_size = max(3, min(10, 5 + gen // 20))

        selected_parents = []
        for _ in range(pop_size):
            tournament_indices = random.sample(range(pop_size), min(tournament_size, pop_size))
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            selected_parents.append(population[winner_idx][:])

        # Elitism: keep the best individual
        new_population = [best_sequence[:]]

        # Generate offspring
        while len(new_population) < pop_size:
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)

            # Crossover
            if len(parent1) > 1 and len(parent2) > 1:
                crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
                child = parent1[:crossover_point] + parent2[crossover_point:]
            else:
                child = parent1[:]

            # Mutation: adaptive based on generation and sequence complexity
            if random.random() < 0.3:
                # Determine mutation strength based on sequence characteristics
                _, conc = get_pattern_score(child)
                mutation_strength = 0.2 if conc < 10 else 0.1
                for i in range(len(child)):
                    if random.random() < 0.1:
                        child[i] *= random.uniform(0.8, 1.2)
                        child[i] = max(0, child[i])

            new_population.append(child)

        population = new_population[:pop_size]

    return best_sequence if best_sequence else adaptive_initialize_sequence()

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
    best_score = 0
    best_sequence = None

    # Multi-start approach: try several initialization strategies
    strategies = [
        lambda: adaptive_initialize_sequence(),
        lambda: generate_random_sequence(),
        lambda: adaptive_initialize_sequence(500),
    ]

    for strategy in strategies:
        sequence = strategy()

        # Apply local improvement
        sequence = local_improvement(sequence, max_iter=20)

        # Try LP-based refinement
        lp_improved = get_good_direction_to_move_into_lp(sequence)
        if lp_improved is not None:
            sequence = lp_improved

        # Evolve using the adaptive strategy
        evolved_seq = adaptive_evolve_sequences()
        evolved_seq = local_improvement(evolved_seq, max_iter=10)
        score = compute_autocorrelation_constant(evolved_seq)

        if score > best_score:
            best_score = score
            best_sequence = evolved_seq[:]

    # Fallback
    if best_sequence is None:
        best_sequence = adaptive_initialize_sequence()

    # Final cleanup
    if len(best_sequence) == 0 or np.sum(best_sequence) < 0.01:
        best_sequence = [1.0]

    # Limit size
    if len(best_sequence) > 1000:
        best_sequence = best_sequence[:1000]

    # Clip values
    best_sequence = [max(0, min(1000, x)) for x in best_sequence]

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")