# EVOLVE-BLOCK-START
import numpy as np
from scipy import signal, optimize
from scipy.fft import fft, ifft
import random
from typing import List, Optional
import time

# Constants
MAX_TIME_SECONDS = 180
MIN_SEQ_LENGTH = 10
MAX_SEQ_LENGTH = 1000
FFT_THRESHOLD = 100  # Use FFT for sequences longer than this
CONVERGENCE_TOLERANCE = 1e-6
MAX_STAGNANT_ITERATIONS = 50
ELITE_SIZE = 5  # Number of top sequences to preserve

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def autocorrelation_constant(sequence: List[float]) -> float:
    """
    Calculates C₁ = 2n * max(b) / (sum(a))^2 where b = a * a (autoconvolution).
    Returns the inverse 1/C₁ which we want to maximize.
    """
    n = len(sequence)
    if n == 0:
        return 0.0

    sum_a = sum(sequence)
    if sum_a < 0.01:
        return 0.0

    # Compute autoconvolution using FFT for efficiency
    if n > FFT_THRESHOLD:
        # Use FFT for fast convolution
        padded_len = 2 * n - 1
        seq_fft = fft(sequence, padded_len)
        conv_fft = seq_fft * seq_fft.conj()  # Element-wise multiplication
        autoconv = ifft(conv_fft).real
        max_conv = max(autoconv)
    else:
        # Direct convolution for small sequences
        autoconv = signal.convolve(sequence, sequence, mode='full')
        max_conv = max(autoconv)

    # Calculate C₁
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    if c1 == 0:
        return 0.0
    return 1.0 / c1

def compute_gradient_approximation(sequence: List[float], epsilon_base: float = 1e-4) -> List[float]:
    """
    Approximate gradient using symmetric finite differences with adaptive epsilon.
    """
    n = len(sequence)
    grad = []
    for i in range(n):
        # Determine adaptive epsilon based on element magnitude
        elem_mag = abs(sequence[i])
        if elem_mag < 1e-6:
            epsilon = epsilon_base
        else:
            epsilon = epsilon_base * elem_mag

        # Perturb dimension i symmetrically
        perturbed_plus = sequence[:]
        perturbed_minus = sequence[:]
        perturbed_plus[i] += epsilon
        perturbed_minus[i] -= epsilon

        # Ensure non-negativity
        perturbed_plus[i] = max(0, perturbed_plus[i])
        perturbed_minus[i] = max(0, perturbed_minus[i])

        # Evaluate function
        f_plus = autocorrelation_constant(perturbed_plus)
        f_minus = autocorrelation_constant(perturbed_minus)

        grad_i = (f_plus - f_minus) / (2 * epsilon)
        grad.append(grad_i)

    return grad

def adaptive_step_size(iteration: int, prev_grad_norm: float = 1.0) -> float:
    """
    Adaptive step size with exponential decay and gradient normalization.
    """
    # Base step size
    base_step = 0.01

    # Exponential decay with iteration count
    decay_rate = 0.98
    base_step *= (decay_rate ** iteration)

    # Adjust based on gradient magnitude
    if prev_grad_norm > 1e-3:
        base_step *= min(1.0, 1.0 / prev_grad_norm)

    # Ensure minimum step size
    base_step = max(base_step, 1e-6)
    return base_step

def optimize_sequence_pso(initial_seq: List[float], max_iter: int = 300) -> List[float]:
    """
    Enhanced Particle Swarm Optimization for finding good initial sequences.
    """
    n = len(initial_seq)
    num_particles = 100  # Increased particle count for better exploration
    max_vel = 0.2  # Increased maximum velocity for faster movement
    inertia = 0.7
    cognitive_weight = 1.5
    social_weight = 1.5

    # Initialize particles with diversified starting points
    particles = []
    velocities = []
    for _ in range(num_particles):
        # Create a more diverse initial population
        particle = [max(0, random.gauss(1, 0.3)) for _ in range(n)]
        particles.append(particle)
        velocities.append([random.uniform(-max_vel, max_vel) for _ in range(n)])

    # Track best positions
    best_positions = [p[:] for p in particles]
    best_scores = [autocorrelation_constant(p) for p in particles]

    # Global best
    global_best_idx = np.argmax(best_scores)
    global_best = best_positions[global_best_idx][:]
    global_best_score = best_scores[global_best_idx]

    # PSO iterations
    for iteration in range(max_iter):
        if time.time() > start_time + MAX_TIME_SECONDS - 2:
            break

        for i in range(num_particles):
            particle = particles[i]
            velocity = velocities[i]

            # Update velocity
            r1, r2 = random.random(), random.random()
            for d in range(n):
                cognitive = cognitive_weight * r1 * (best_positions[i][d] - particle[d])
                social = social_weight * r2 * (global_best[d] - particle[d])
                velocity[d] = inertia * velocity[d] + cognitive + social

                # Clamp velocity
                velocity[d] = max(-max_vel, min(max_vel, velocity[d]))

                # Update position
                particle[d] += velocity[d]
                particle[d] = max(0, particle[d])  # Enforce non-negativity

            # Evaluate new position
            score = autocorrelation_constant(particle)

            # Update personal best
            if score > best_scores[i]:
                best_scores[i] = score
                best_positions[i] = particle[:]

                # Update global best
                if score > global_best_score:
                    global_best_score = score
                    global_best = particle[:]

    return global_best

def get_good_direction_to_move_into(sequence: List[float]) -> Optional[List[float]]:
    """
    Returns the direction to move into the sequence using enhanced gradient ascent and PSO.
    """
    start_time = time.time()
    n = len(sequence)
    if n == 0:
        return None

    # Check if we have enough elements
    if n < MIN_SEQ_LENGTH:
        # Expand sequence
        extended_seq = sequence + [0.0] * (MIN_SEQ_LENGTH - n)
        sequence = extended_seq

    # Use PSO to find a good starting point
    try:
        sequence = optimize_sequence_pso(sequence, 300)
    except Exception as e:
        pass  # Fallback to regular sequence

    # Perform enhanced gradient ascent with elite preservation
    current_inv_c1 = autocorrelation_constant(sequence)
    prev_grad_norm = 1.0
    stagnant_count = 0

    # Elite sequences to preserve good solutions
    elite_sequences = []
    elite_scores = []

    # Maximum number of gradient steps
    max_steps = 1000
    for step in range(max_steps):
        if time.time() > start_time + MAX_TIME_SECONDS - 2:
            break

        # Compute gradient
        try:
            gradient = compute_gradient_approximation(sequence)
        except Exception:
            return None

        # Normalize gradient
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < 1e-10:
            break

        # Compute adaptive step size with exponential decay
        step_size = adaptive_step_size(step, prev_grad_norm)

        # Update sequence
        new_sequence = []
        for i in range(len(sequence)):
            new_val = sequence[i] + step_size * gradient[i]
            new_sequence.append(max(0, new_val))

        # Evaluate new sequence
        new_inv_c1 = autocorrelation_constant(new_sequence)

        if new_inv_c1 > current_inv_c1:
            sequence = new_sequence
            current_inv_c1 = new_inv_c1
            stagnant_count = 0

            # Preserve elite sequences
            if len(elite_sequences) < ELITE_SIZE:
                elite_sequences.append(sequence[:])
                elite_scores.append(current_inv_c1)
            else:
                # Replace the worst elite if new is better
                worst_idx = np.argmin(elite_scores)
                if current_inv_c1 > elite_scores[worst_idx]:
                    elite_sequences[worst_idx] = sequence[:]
                    elite_scores[worst_idx] = current_inv_c1
        else:
            stagnant_count += 1
            if stagnant_count > MAX_STAGNANT_ITERATIONS:
                # Occasionally restart from elite if stuck
                if elite_sequences:
                    restart_idx = np.argmax(elite_scores)
                    sequence = elite_sequences[restart_idx][:]
                    current_inv_c1 = elite_scores[restart_idx]
                break

        prev_grad_norm = grad_norm

    return sequence

def search_for_best_sequence() -> List[float]:
    """
    Function to search for the best coefficient sequence.
    """
    global start_time
    start_time = time.time()

    # Start with a random sequence of medium length
    n = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
    sequence = [random.uniform(0.1, 1.0) for _ in range(n)]

    # Try to improve it
    improved_sequence = get_good_direction_to_move_into(sequence)

    if improved_sequence is not None and len(improved_sequence) > 0:
        return improved_sequence
    else:
        # Return original if nothing works
        return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")