# EVOLVE-BLOCK-START
import numpy as np
from scipy.fft import fft, ifft
import random
import time

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_convolution_fft(seq):
    """Compute the autoconvolution using FFT for efficiency."""
    n = len(seq)
    padded_seq = np.pad(seq, (0, n-1), mode='constant')
    conv_result = ifft(fft(padded_seq) * np.conj(fft(padded_seq)))
    return np.real(conv_result[:n])

def calculate_c1(sequence):
    """Calculate the C1 constant for a given sequence."""
    if len(sequence) == 0:
        return float('inf')

    sequence = np.array(sequence)
    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return float('inf')

    # Use FFT-based convolution for efficiency
    conv = compute_convolution_fft(sequence)
    max_b = np.max(conv)
    n = len(sequence)

    # Avoid division by zero or very small numbers
    if max_b <= 1e-12:
        return float('inf')

    c1 = (2 * n * max_b) / (sum_a ** 2)
    return c1

def evaluate_fitness(sequence):
    """Evaluate the inverse of C1 as fitness (we want to maximize 1/C1)."""
    c1 = calculate_c1(sequence)
    if c1 == float('inf'):
        return 0.0  # Penalty for invalid sequences
    return 1.0 / c1

def compute_convolution_gradient_fft(sequence):
    """Compute gradient of maximum convolution value w.r.t. sequence elements using FFT."""
    n = len(sequence)
    if n == 0:
        return np.zeros(n)

    # Compute base convolution using FFT
    conv = compute_convolution_fft(sequence)
    max_conv_idx = np.argmax(conv)
    max_conv_val = conv[max_conv_idx]

    # Compute gradient using finite difference with FFT for efficiency
    eps = 1e-6
    grad = np.zeros(n)
    
    # For each element, compute gradient using convolution differences
    for i in range(n):
        # Create perturbed sequence
        perturbed_seq = sequence.copy()
        perturbed_seq[i] += eps
        
        # Compute convolution for perturbed sequence
        perturbed_conv = compute_convolution_fft(perturbed_seq)
        perturbed_max_conv = np.max(perturbed_conv)
        
        # Compute gradient component
        grad[i] = (perturbed_max_conv - max_conv_val) / eps

    return grad

def gradient_ascent_step(current_sequence, learning_rate=0.01):
    """Perform a single gradient ascent step."""
    # Compute gradient
    grad = compute_convolution_gradient_fft(current_sequence)
    
    # Update using gradient ascent
    new_sequence = current_sequence + learning_rate * grad
    
    # Ensure non-negativity
    new_sequence = np.maximum(new_sequence, 0)
    
    # Ensure minimum sum
    if np.sum(new_sequence) < 0.01:
        new_sequence[0] = max(0.1, new_sequence[0])
    
    # Clip to [0, 1000]
    new_sequence = np.clip(new_sequence, 0, 1000)
    
    return new_sequence

def multi_resolution_optimization(initial_sequence, max_iterations=100):
    """Optimize using multi-resolution approach."""
    sequence = np.array(initial_sequence)
    best_fitness = evaluate_fitness(sequence)
    best_sequence = sequence.copy()
    
    # Coarse resolution optimization
    lr_coarse = 0.1
    for _ in range(5):
        sequence = gradient_ascent_step(sequence, lr_coarse)
        fitness = evaluate_fitness(sequence)
        if fitness > best_fitness:
            best_fitness = fitness
            best_sequence = sequence.copy()
    
    # Medium resolution optimization
    lr_medium = 0.05
    for _ in range(10):
        sequence = gradient_ascent_step(sequence, lr_medium)
        fitness = evaluate_fitness(sequence)
        if fitness > best_fitness:
            best_fitness = fitness
            best_sequence = sequence.copy()
    
    # Fine resolution optimization
    lr_fine = 0.01
    for i in range(max_iterations):
        sequence = gradient_ascent_step(sequence, lr_fine)
        fitness = evaluate_fitness(sequence)
        if fitness > best_fitness:
            best_fitness = fitness
            best_sequence = sequence.copy()
        
        # Adaptive learning rate decay
        lr_fine *= 0.995
    
    return best_sequence.tolist()

def generate_smart_initial_sequence():
    """Generate an intelligent initial sequence based on theoretical expectations."""
    n = random.randint(100, 1000)
    
    # Combine multiple patterns for better exploration
    patterns = []
    
    # Power law decay
    alpha = 0.95
    power_law = [alpha ** i for i in range(n)]
    power_law = np.array(power_law) / np.sum(power_law) * 1000
    
    # Linear decay
    linear_decay = np.array([1.0 - i/(n-1) if n > 1 else 1.0 for i in range(n)])
    linear_decay = linear_decay / np.sum(linear_decay) * 1000
    
    # Random noise
    noise = np.random.uniform(0, 100, n)
    
    # Combine patterns
    combined = (power_law + linear_decay + noise) / 3
    
    # Ensure non-negativity and minimum sum
    combined = np.maximum(combined, 0)
    if np.sum(combined) < 0.01:
        combined[0] = 100
    
    return combined.tolist()

def adaptive_local_search(initial_sequence, max_iter=200):
    """Perform adaptive local search with dynamic parameters."""
    current_sequence = np.array(initial_sequence)
    current_fitness = evaluate_fitness(current_sequence)
    best_sequence = current_sequence.copy()
    best_fitness = current_fitness
    
    # Dynamic parameters
    temp = 1.0
    cooling_rate = 0.95
    min_temp = 1e-4
    
    for iteration in range(max_iter):
        # Perturb sequence
        mutated = current_sequence.copy()
        for i in range(len(mutated)):
            if random.random() < 0.3:
                delta = np.random.normal(0, 0.1 * np.mean(mutated) if np.mean(mutated) > 0 else 0.1)
                mutated[i] = max(0.0, mutated[i] + delta)
        
        # Clip to valid range
        mutated = np.clip(mutated, 0, 1000)
        
        mutated_fitness = evaluate_fitness(mutated)
        
        # Accept or reject based on simulated annealing criteria
        if mutated_fitness > current_fitness or random.random() < np.exp((mutated_fitness - current_fitness) / (temp + 1e-10)):
            current_sequence = mutated
            current_fitness = mutated_fitness
        
        # Update best
        if current_fitness > best_fitness:
            best_sequence = current_sequence.copy()
            best_fitness = current_fitness
        
        # Cool down temperature
        temp = max(temp * cooling_rate, min_temp)
    
    return best_sequence.tolist(), best_fitness

def search_for_best_sequence():
    """Main search function using the new fast convolution gradient ascent approach."""
    start_time = time.time()
    best_inv_c1 = 0.0
    best_sequence = None
    
    # Multiple initialization attempts with smart sequences
    for attempt in range(30):
        if time.time() - start_time > 170:
            break
            
        # Generate smart initial sequence
        init_sequence = generate_smart_initial_sequence()
        
        # Multi-resolution optimization
        optimized_sequence = multi_resolution_optimization(init_sequence, max_iterations=30)
        
        # Evaluate optimized sequence
        inv_c1 = evaluate_fitness(optimized_sequence)
        
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = optimized_sequence[:]
    
    # If no good sequence found, fallback to a basic approach
    if best_sequence is None:
        fallback_sequence = generate_smart_initial_sequence()
        best_sequence = multi_resolution_optimization(fallback_sequence, max_iterations=50)
    
    # Final refinement with adaptive local search
    refined_sequence, refined_fitness = adaptive_local_search(best_sequence, 100)
    if refined_fitness > best_inv_c1:
        best_inv_c1 = refined_fitness
        best_sequence = refined_sequence
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")