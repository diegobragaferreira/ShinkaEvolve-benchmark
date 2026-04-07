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

def generate_step_function_sequence(length=None, step_count=None, pattern_type="exponential"):
    """Generate step function sequences with specific pattern types."""
    if length is None:
        length = random.randint(100, 1000)
    if step_count is None:
        step_count = random.randint(5, 20)
    
    # Determine step sizes
    step_sizes = []
    for i in range(step_count):
        step_sizes.append(length // step_count + (1 if i < length % step_count else 0))
    
    # Create step function based on pattern type
    sequence = []
    if pattern_type == "exponential":
        # Exponential decay of step heights
        base_height = 1000
        decay_factor = 0.9
        heights = [base_height * (decay_factor ** i) for i in range(step_count)]
        total_height = sum(heights)
        if total_height > 0:
            heights = [h / total_height * 100 for h in heights]
        
        for i, size in enumerate(step_sizes):
            sequence.extend([heights[i]] * size)
            
    elif pattern_type == "periodic":
        # Alternating high/low steps
        heights = []
        for i in range(step_count):
            if i % 2 == 0:
                heights.append(1000)
            else:
                heights.append(100)
        
        total_height = sum(heights)
        if total_height > 0:
            heights = [h / total_height * 100 for h in heights]
        
        for i, size in enumerate(step_sizes):
            sequence.extend([heights[i]] * size)
            
    elif pattern_type == "uniform":
        # Uniform heights
        height = 1000 / step_count
        for i, size in enumerate(step_sizes):
            sequence.extend([height] * size)
            
    else:
        # Random heights
        heights = [random.uniform(0, 1000) for _ in range(step_count)]
        total_height = sum(heights)
        if total_height > 0:
            heights = [h / total_height * 100 for h in heights]
        
        for i, size in enumerate(step_sizes):
            sequence.extend([heights[i]] * size)
    
    return sequence

def local_step_refinement(sequence, iterations=10):
    """
    Apply local refinement to step function by adjusting step boundaries and heights.
    """
    sequence = sequence[:]
    best_score = compute_autocorrelation_constant(sequence)
    best_seq = sequence[:]
    
    for _ in range(iterations):
        # Try to adjust step boundaries slightly
        new_seq = sequence[:]
        idx = random.randint(0, len(new_seq) - 1)
        # Perturb nearby elements
        for i in range(max(0, idx - 3), min(len(new_seq), idx + 4)):
            if random.random() < 0.3:
                new_seq[i] = max(0, min(1000, new_seq[i] + random.uniform(-50, 50)))
        
        # Evaluate and potentially accept change
        new_score = compute_autocorrelation_constant(new_seq)
        if new_score > best_score:
            best_score = new_score
            best_seq = new_seq[:]
            sequence = new_seq[:]
    
    return best_seq

def search_for_best_sequence() -> list[float]:
    """Main function to find the best sequence using step function optimizations."""
    start_time = time.time()
    
    best_score = 0.0
    best_sequence = []
    
    # Strategy 1: Exponential decay step functions
    for _ in range(10):
        seq = generate_step_function_sequence(pattern_type="exponential")
        seq = local_step_refinement(seq, 15)
        score = compute_autocorrelation_constant(seq)
        if score > best_score:
            best_score = score
            best_sequence = seq[:]
            
        if time.time() - start_time > 170:
            return best_sequence
    
    # Strategy 2: Periodic step functions
    for _ in range(10):
        seq = generate_step_function_sequence(pattern_type="periodic")
        seq = local_step_refinement(seq, 15)
        score = compute_autocorrelation_constant(seq)
        if score > best_score:
            best_score = score
            best_sequence = seq[:]
            
        if time.time() - start_time > 170:
            return best_sequence
    
    # Strategy 3: Uniform step functions
    for _ in range(10):
        seq = generate_step_function_sequence(pattern_type="uniform")
        seq = local_step_refinement(seq, 15)
        score = compute_autocorrelation_constant(seq)
        if score > best_score:
            best_score = score
            best_sequence = seq[:]
            
        if time.time() - start_time > 170:
            return best_sequence
    
    # Strategy 4: Random step functions
    for _ in range(10):
        seq = generate_step_function_sequence(pattern_type="random")
        seq = local_step_refinement(seq, 15)
        score = compute_autocorrelation_constant(seq)
        if score > best_score:
            best_score = score
            best_sequence = seq[:]
            
        if time.time() - start_time > 170:
            return best_sequence
    
    # Final refinement of the best found sequence
    if best_sequence:
        best_sequence = local_step_refinement(best_sequence, 20)
        best_score = compute_autocorrelation_constant(best_sequence)
    
    # Fallback to simple sequence if needed
    if not best_sequence or best_score < 0.01:
        best_sequence = [1.0] * 100
    
    # Final verification and cleanup
    if len(best_sequence) == 0 or np.sum(best_sequence) < 0.01:
        best_sequence = [1.0]
    
    # Limit size to prevent excessive computation
    if len(best_sequence) > 1000:
        best_sequence = best_sequence[:1000]
    
    # Clip values to [0, 1000] for practicality
    best_sequence = [max(0, min(1000, x)) for x in best_sequence]
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")