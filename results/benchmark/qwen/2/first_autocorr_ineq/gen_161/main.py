# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
import random
import time
from collections import deque
import math

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_c1(sequence):
    """Compute C1 for a given sequence."""
    if len(sequence) == 0 or abs(sum(sequence)) < 1e-10:
        return float('inf')
    convolved = fftconvolve(sequence, sequence, mode='full')
    max_conv = np.max(convolved)
    sum_seq = sum(sequence)
    return (2 * len(sequence) * max_conv) / (sum_seq * sum_seq)

def evaluate_sequence(sequence):
    """Evaluate a sequence by computing 1/C1."""
    c1 = compute_c1(sequence)
    if c1 == float('inf') or c1 > 1e10:
        return float('-inf')
    return 1.0 / c1

def generate_structured_sequence(length):
    """Generate a structured sequence with good properties."""
    base_sequence = np.random.uniform(0, 100, length)
    
    if np.random.random() < 0.5:
        idxs = np.random.choice(length, size=min(10, length//4), replace=False)
        base_sequence[idxs] *= np.random.uniform(5, 20)
    
    if np.random.random() < 0.3:
        threshold = np.random.choice(length)
        base_sequence[threshold:] = 0
    
    return base_sequence.tolist()

def generate_uniform_sequence(length):
    """Generate a uniform sequence."""
    return [1.0] * length

def generate_step_sequence(length):
    """Generate a step sequence."""
    mid = length // 2
    return [1.0] * mid + [0.0] * (length - mid)

def initialize_sequences(count=10, min_length=100, max_length=1000):
    """Initialize a diverse set of sequences."""
    sequences = []
    
    # Add known good structures
    sequences.append(generate_uniform_sequence(100))
    sequences.append(generate_step_sequence(100))
    
    # Add random structured sequences
    for _ in range(count - 2):
        n = random.randint(min_length, max_length)
        seq = generate_structured_sequence(n)
        sequences.append(seq)
    
    return sequences

def hamiltonian_dynamics(q, p, grad_U, step_size=0.01, num_steps=10):
    """
    Perform Hamiltonian dynamics using leapfrog integration.
    q: positions (sequence values)
    p: momenta
    grad_U: gradient of potential (negative of gradient of 1/C1)
    """
    # Copy initial values
    q_new = q.copy()
    p_new = p.copy()
    
    # Leapfrog integration
    for _ in range(num_steps):
        # Update momentum
        p_new = [p_new[i] - step_size * grad_U[i] for i in range(len(p_new))]
        # Update position
        q_new = [q_new[i] + step_size * p_new[i] for i in range(len(q_new))]
    
    return q_new, p_new

def potential_energy(sequence):
    """Potential energy corresponding to 1/C1 (negative log likelihood)."""
    inv_c1 = evaluate_sequence(sequence)
    # Convert to negative log likelihood for HMC
    if inv_c1 == float('-inf'):
        return float('inf')
    return -math.log(max(inv_c1, 1e-100))  # Prevent log(0)

def gradient_of_potential(sequence):
    """Approximate gradient of potential using finite differences."""
    epsilon = 1e-6
    grad = []
    for i in range(len(sequence)):
        # Perturb dimension i
        seq_plus = sequence.copy()
        seq_minus = sequence.copy()
        seq_plus[i] += epsilon
        seq_minus[i] -= epsilon
        
        # Evaluate potential
        U_plus = potential_energy(seq_plus)
        U_minus = potential_energy(seq_minus)
        
        # Gradient approximation
        grad.append((U_minus - U_plus) / (2 * epsilon))
    
    return grad

def hamiltonian_monte_carlo(sequence, num_samples=100, step_size=0.01, num_leapfrog=10):
    """
    Perform Hamiltonian Monte Carlo sampling to optimize sequence.
    """
    current_sequence = sequence.copy()
    current_energy = potential_energy(current_sequence)
    
    accepted = 0
    total_proposals = 0
    
    for _ in range(num_samples):
        # Sample momentum
        momenta = [np.random.normal(0, 1) for _ in range(len(current_sequence))]
        
        # Propose new position using Hamiltonian dynamics
        proposed_sequence, proposed_momenta = hamiltonian_dynamics(
            current_sequence, momenta, 
            gradient_of_potential(current_sequence),
            step_size, num_leapfrog
        )
        
        # Metropolis acceptance step
        proposed_energy = potential_energy(proposed_sequence)
        delta_energy = proposed_energy - current_energy
        
        if delta_energy < 0 or np.random.random() < np.exp(-delta_energy):
            current_sequence = proposed_sequence
            current_energy = proposed_energy
            accepted += 1
        
        total_proposals += 1
    
    return current_sequence

def search_for_best_sequence():
    """Main search function implementing Hamiltonian Monte Carlo optimization."""
    start_time = time.time()
    max_time_seconds = 170
    
    # Initialize diverse sequences
    sequences = initialize_sequences()
    best_sequence = None
    best_inv_c1 = float('-inf')
    
    # History tracking for stagnation detection
    history = deque(maxlen=10)
    
    for attempt in range(20):  # Multiple attempts
        if time.time() - start_time > max_time_seconds:
            break
        
        # Select a random initial sequence
        current_sequence = random.choice(sequences)
        
        # HMC optimization loop
        for iteration in range(50):  # Fewer iterations to allow more sampling
            if time.time() - start_time > max_time_seconds:
                break
            
            # Perform HMC sampling
            optimized_sequence = hamiltonian_monte_carlo(
                current_sequence, 
                num_samples=20, 
                step_size=0.01, 
                num_leapfrog=5
            )
            
            # Evaluate optimized sequence
            new_inv_c1 = evaluate_sequence(optimized_sequence)
            
            # Accept improvement
            if new_inv_c1 > best_inv_c1:
                best_inv_c1 = new_inv_c1
                best_sequence = optimized_sequence.copy()
            
            current_sequence = optimized_sequence
            
            # Track history
            history.append(new_inv_c1)
            
            # Stagnation detection and reset
            if len(history) == history.maxlen:
                recent_change = abs(history[-1] - history[0])
                if recent_change < 1e-6:
                    current_sequence = generate_structured_sequence(len(current_sequence))
    
    # Final evaluation of best sequence
    if best_sequence is not None:
        final_inv_c1 = evaluate_sequence(best_sequence)
        if final_inv_c1 > best_inv_c1:
            best_inv_c1 = final_inv_c1
    
    # Fallback
    if best_sequence is None:
        best_sequence = [1.0] * 100
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
