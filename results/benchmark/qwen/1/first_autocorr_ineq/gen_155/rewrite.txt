# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.signal import fftconvolve
import time

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

def create_initial_sequence():
    """Create an initial sequence inspired by mathematical structures known to yield good results."""
    # Use a sequence with decreasing heights to balance mass and reduce peak convolution
    length = random.randint(100, 1000)
    sequence = [1000 / (1 + i/10.0) for i in range(length)]
    
    # Normalize to have reasonable total mass
    total_mass = sum(sequence)
    if total_mass > 0:
        sequence = [x / total_mass * 100 for x in sequence]
        
    return sequence

def local_improve_sequence(sequence, max_iterations=100):
    """
    Apply local improvement using gradient-based method.
    """
    current_seq = sequence[:]
    best_score = compute_autocorrelation_constant(current_seq)
    best_seq = current_seq[:]

    for _ in range(max_iterations):
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

def search_for_best_sequence() -> list[float]:
    """
    Main function to find the best coefficient sequence.
    Uses a more direct gradient ascent approach from a carefully constructed initial sequence.
    """
    start_time = time.time()
    best_score = 0
    best_sequence = None

    # Try multiple initializations to escape local optima
    for _ in range(20):
        # Start with a structured initial sequence
        sequence = create_initial_sequence()
        
        # Improve locally
        sequence = local_improve_sequence(sequence, max_iterations=50)
        
        # Evaluate
        score = compute_autocorrelation_constant(sequence)
        if score > best_score:
            best_score = score
            best_sequence = sequence[:]

    # Fallback if nothing worked
    if best_sequence is None:
        best_sequence = [1.0] * 100

    # Final verification
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