# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
import time
import math

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

def generate_random_sequence(length=None, min_length=10, max_length=1000):
    """Generate a random sequence with specified or random length."""
    if length is None:
        length = random.randint(min_length, max_length)
    sequence = [random.uniform(0, 1000) for _ in range(length)]
    if all(x == 0 for x in sequence):
        sequence[0] = 1.0
    return sequence

def adaptive_hill_climb(initial_sequence, max_iter=1000):
    """
    Adaptive hill climbing with dynamic step size and convex relaxation.
    """
    current_sequence = np.array(initial_sequence, dtype=float)
    current_fitness = evaluate_sequence(current_sequence)
    
    # Track recent improvements for adaptive step sizing
    recent_improvements = []
    max_recent = 5
    
    # Initial step size and decay factor
    step_size = 0.01
    decay_factor = 0.99
    
    for iteration in range(max_iter):
        # Estimate gradient via finite differences
        eps = 1e-4
        gradient = np.zeros_like(current_sequence)
        base_fitness = current_fitness
        
        # Compute gradient for each dimension
        for i in range(len(current_sequence)):
            perturbed = current_sequence.copy()
            perturbed[i] += eps
            perturbed_fitness = evaluate_sequence(perturbed)
            gradient[i] = (perturbed_fitness - base_fitness) / eps
        
        # Use convex relaxation to find direction
        direction = -gradient  # Negative gradient for ascent
        direction_normalized = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Adjust step size adaptively based on recent performance
        if len(recent_improvements) > 0:
            avg_improvement = np.mean(recent_improvements)
            if avg_improvement > 0:
                step_size *= 1.05  # Increase step if improving
            else:
                step_size *= 0.95  # Decrease step if not improving
                step_size = max(step_size, 1e-6)  # Minimum step size
        
        # Take step in direction
        new_sequence = current_sequence + step_size * direction_normalized
        new_sequence = np.clip(new_sequence, 0, 1000)  # Clip to valid range
        
        # Evaluate new position
        new_fitness = evaluate_sequence(new_sequence)
        
        # Accept improvement
        if new_fitness > current_fitness:
            current_sequence = new_sequence
            current_fitness = new_fitness
            recent_improvements.append(new_fitness - current_fitness)
        else:
            # Occasionally accept worse moves to escape local maxima
            if random.random() < 0.01:
                current_sequence = new_sequence
                current_fitness = new_fitness
                recent_improvements.append(new_fitness - current_fitness)
        
        # Maintain recent improvements window
        if len(recent_improvements) > max_recent:
            recent_improvements.pop(0)
        
        # Adaptive decay of step size
        step_size *= decay_factor
        
        # Early stopping if no improvement recently
        if len(recent_improvements) >= 3 and np.mean(recent_improvements[-3:]) < 1e-8:
            break
    
    return current_sequence.tolist(), current_fitness

def convex_direction_search(sequence, max_search_steps=20):
    """
    Uses a simple convex relaxation to find a better direction.
    This is a simplified version that finds a descent direction without full LP.
    """
    n = len(sequence)
    if n < 2:
        return sequence
    
    # Create a simple convex optimization model for guidance
    # We'll approximate the gradient and use a descent direction
    eps = 1e-4
    base_fitness = evaluate_sequence(sequence)
    
    # Estimate gradient
    gradient = np.zeros_like(sequence)
    for i in range(n):
        perturbed = sequence.copy()
        perturbed[i] += eps
        perturbed_fitness = evaluate_sequence(perturbed)
        gradient[i] = (perturbed_fitness - base_fitness) / eps
    
    # Use a simple descent direction (negative gradient)
    # Normalize for unit step
    norm_grad = np.linalg.norm(gradient)
    if norm_grad > 0:
        descent_dir = -gradient / norm_grad
    else:
        descent_dir = np.zeros_like(sequence)
    
    # Take a small step to test
    step_size = 0.01
    candidate = sequence + step_size * descent_dir
    candidate = np.clip(candidate, 0, 1000)
    
    # Accept if better, otherwise return original
    if evaluate_sequence(candidate) > base_fitness:
        return candidate.tolist()
    else:
        return sequence

def search_for_best_sequence():
    """Main search function to find the best sequence."""
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Try multiple starting points
    for attempt in range(10):
        # Generate a random initial sequence
        initial_sequence = generate_random_sequence()
        
        # Try adaptive hill climb
        climb_seq, climb_fitness = adaptive_hill_climb(initial_sequence, 500)
        if climb_fitness > best_inv_c1:
            best_inv_c1 = climb_fitness
            best_sequence = climb_seq
            
        # Also try convex direction search
        conv_seq = convex_direction_search(initial_sequence)
        conv_fitness = evaluate_sequence(conv_seq)
        if conv_fitness > best_inv_c1:
            best_inv_c1 = conv_fitness
            best_sequence = conv_seq
            
        # Try another adaptive climb with a different starting point
        alt_seq = generate_random_sequence()
        alt_seq, alt_fitness = adaptive_hill_climb(alt_seq, 300)
        if alt_fitness > best_inv_c1:
            best_inv_c1 = alt_fitness
            best_sequence = alt_seq
    
    # Final refinement with adaptive climb on best found
    if best_sequence is not None:
        ref_seq, ref_fitness = adaptive_hill_climb(best_sequence, 500)
        if ref_fitness > best_inv_c1:
            best_inv_c1 = ref_fitness
            best_sequence = ref_seq
    
    # Return best found or default
    if best_sequence is None:
        best_sequence = generate_random_sequence()
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")