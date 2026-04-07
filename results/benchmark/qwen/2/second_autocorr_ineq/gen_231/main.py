# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import time
from numba import jit, prange
import random
from typing import List, Tuple

@jit(nopython=True)
def compute_autoconvolution_numba(f):
    """Numba-accelerated autoconvolution computation"""
    n = len(f)
    g = np.zeros(2*n - 1)
    
    # Manual convolution loop for speed
    for i in range(n):
        for j in range(n):
            g[i + j] += f[i] * f[j]
    
    return g[n-1:]  # Return positive lags only

@jit(nopython=True)
def compute_norms_numba(g):
    """Numba-accelerated norm computations"""
    n = len(g)
    
    # Compute norms
    norm_1 = 0.0
    norm_2_sq = 0.0
    norm_inf = 0.0
    
    for i in range(n):
        abs_g = abs(g[i])
        norm_1 += abs_g
        norm_2_sq += abs_g * abs_g
        if abs_g > norm_inf:
            norm_inf = abs_g
    
    return norm_1, norm_2_sq, norm_inf

@jit(nopython=True)
def compute_c2_numba(norm_1, norm_2_sq, norm_inf):
    """Numba-accelerated C2 computation"""
    if norm_1 < 1e-12 or norm_inf < 1e-12:
        return 0.0
    return norm_2_sq / (norm_1 * norm_inf)

def evaluate_function(f):
    """Evaluate the function and compute C2"""
    try:
        # Fast autoconvolution
        g = compute_autoconvolution_numba(f)
        
        # Fast norm computations
        norm_1, norm_2_sq, norm_inf = compute_norms_numba(g)
        
        # C2 computation
        c2 = compute_c2_numba(norm_1, norm_2_sq, norm_inf)
        
        return c2, g
    except Exception:
        return 0.0, np.array([0.0])

def spectral_peak_optimizer(n_steps: int = 1000) -> List[float]:
    """
    Spectral-based peak optimizer that creates optimal step functions
    by designing functions with specific frequency characteristics
    that maximize C2 = ||g||₂² / (||g||₁ · ||g||∞)
    """
    # Domain setup
    domain_width = 0.5
    domain_center = 0.0
    
    # Create a structured function based on mathematical principles
    # We want to maximize ||g||₂² while minimizing ||g||₁ · ||g||∞
    # This suggests creating a function with relatively flat autoconvolution
    
    # Phase 1: Design frequency spectrum that produces favorable autoconvolution
    x = np.linspace(-domain_width/2, domain_width/2, n_steps)
    
    # Use a combination of carefully chosen sinusoidal components
    # The idea is to create a function whose autoconvolution has minimal sharp peaks
    # This is done by combining components that create constructive interference
    # in the convolution while maintaining overall smoothness
    
    # Base function with multiple harmonics
    base_func = np.zeros(n_steps)
    
    # Primary frequency components selected to create beneficial autoconvolution properties
    frequencies = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]  # Different frequencies
    amplitudes = [1.0, 0.8, 0.7, 0.5, 0.3, 0.2]   # Decreasing amplitudes
    
    # Create the base function using cosine waves (symmetric)
    for freq, amp in zip(frequencies, amplitudes):
        base_func += amp * np.cos(freq * np.pi * x / (domain_width/2))
    
    # Phase 2: Add structured randomness to break degeneracy 
    # but maintain mathematical properties
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, n_steps)
    base_func += noise
    
    # Phase 3: Apply mathematical constraints to improve C2
    # Clip negative values (as required by problem)
    base_func = np.maximum(base_func, 0)
    
    # Scale to reasonable magnitude
    if np.max(base_func) > 0:
        base_func = base_func / np.max(base_func) * 50
    
    # Convert to list format
    return base_func.tolist()

def analytical_peak_design(n_steps: int = 1000) -> List[float]:
    """
    Analytical approach based on mathematical optimization of peak characteristics.
    This creates a function where the autoconvolution has maximal flatness.
    """
    # Mathematical approach to maximize C2
    domain_width = 0.5
    x = np.linspace(-domain_width/2, domain_width/2, n_steps)
    
    # Create function using logarithmic peak placement combined with 
    # geometric progression of amplitudes for optimal autoconvolution properties
    n_peaks = 8
    
    # Use golden-ratio based placement for even distribution
    phi = (1 + np.sqrt(5)) / 2
    peak_positions = []
    
    # Generate peak positions using golden ratio for optimal spacing
    for i in range(n_peaks):
        ratio = (i * phi) % 1.0
        # Map to [-0.25, 0.25] with sine transform for better distribution
        pos = domain_width/2 * np.sin(ratio * np.pi)
        if i % 2 == 1:  # Alternate sides
            pos = -pos
        peak_positions.append(pos)
    
    # Create peaks with geometrically decreasing amplitudes
    peak_params = []
    base_amplitude = 30.0  # Starting amplitude
    
    for i, pos in enumerate(peak_positions):
        # Geometric decay of amplitudes
        amplitude = base_amplitude * (0.8 ** i)
        width = 0.03 + np.random.random() * 0.02  # Slightly varied widths
        peak_params.extend([amplitude, pos, width])
    
    # Generate function using the peaks
    result = np.zeros_like(x)
    for i in range(0, len(peak_params), 3):
        amp, center, width = peak_params[i], peak_params[i+1], peak_params[i+2]
        if width > 1e-6:  # Avoid numerical issues
            result += amp * np.exp(-0.5 * ((x - center) / width)**2)
    
    # Ensure non-negative
    result = np.maximum(result, 0)
    
    # Normalize to reasonable scale
    if np.max(result) > 0:
        result = result / np.max(result) * 40
    
    return result.tolist()

def iterative_refinement(f_values: List[float], max_iterations: int = 200) -> List[float]:
    """
    Advanced iterative refinement using gradient-like updates
    guided by mathematical properties of C2 optimization
    """
    current_f = list(f_values)
    current_c2, _ = evaluate_function(current_f)
    
    # Track progress for early stopping
    prev_c2 = current_c2
    stagnation_counter = 0
    max_stagnation = 20
    
    # Adaptive learning rates
    learning_rate = 0.05
    decay_rate = 0.99
    
    for iteration in range(max_iterations):
        # Create candidate by making small modifications
        candidate_f = list(current_f)
        
        # Modify a fraction of the values more aggressively
        n_modify = max(1, len(candidate_f) // 20)
        indices_to_modify = np.random.choice(len(candidate_f), n_modify, replace=False)
        
        for idx in indices_to_modify:
            # Calculate gradient approximation using finite differences
            dx = 0.1
            original_val = candidate_f[idx]
            
            # Perturb upward and downward
            candidate_f[idx] = max(0, original_val + dx)
            c2_up, _ = evaluate_function(candidate_f)
            
            candidate_f[idx] = max(0, original_val - dx)
            c2_down, _ = evaluate_function(candidate_f)
            
            # Estimate gradient
            grad = (c2_up - c2_down) / (2 * dx)
            
            # Update using gradient with adaptive learning rate
            candidate_f[idx] = max(0, original_val + learning_rate * grad)
        
        # Evaluate candidate
        candidate_c2, _ = evaluate_function(candidate_f)
        
        # Accept improvement or with some probability (simulated annealing)
        if candidate_c2 > current_c2:
            current_f = candidate_f
            current_c2 = candidate_c2
            stagnation_counter = 0
        else:
            # Accept worse solution with probability based on difference and learning rate
            if np.random.random() < np.exp((candidate_c2 - current_c2) / (learning_rate + 1e-8)):
                current_f = candidate_f
                current_c2 = candidate_c2
            stagnation_counter += 1
        
        # Adjust learning rate
        learning_rate *= decay_rate
        
        # Early stopping conditions
        if stagnation_counter > max_stagnation:
            break
            
        if abs(current_c2 - prev_c2) < 1e-6:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            
        prev_c2 = current_c2
    
    return current_f

def construct_function() -> list[float]:
    """
    Main function that constructs a step function to maximize C2.
    Uses mathematical insights from spectral analysis and optimization.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Parameters
    max_time = 90.0  # seconds
    start_time = time.time()
    
    # Initialize best solution
    best_c2 = 0.0
    best_f = None
    
    # Strategy 1: Spectral peak optimization
    try:
        spectral_func = spectral_peak_optimizer(1000)
        c2_spectral, _ = evaluate_function(spectral_func)
        
        if c2_spectral > best_c2:
            best_c2 = c2_spectral
            best_f = spectral_func.copy()
    except Exception:
        pass
    
    # Strategy 2: Analytical peak design
    try:
        analytical_func = analytical_peak_design(1000)
        c2_analytical, _ = evaluate_function(analytical_func)
        
        if c2_analytical > best_c2:
            best_c2 = c2_analytical
            best_f = analytical_func.copy()
    except Exception:
        pass
    
    # Strategy 3: Iterative refinement of best candidates
    if best_f is not None:
        try:
            refined_f = iterative_refinement(best_f, max_iterations=150)
            c2_refined, _ = evaluate_function(refined_f)
            
            if c2_refined > best_c2:
                best_c2 = c2_refined
                best_f = refined_f.copy()
        except Exception:
            pass
    
    # Strategy 4: Hybrid approach combining multiple strategies
    if best_c2 < 0.8:
        try:
            # Create a more sophisticated hybrid function
            domain_width = 0.5
            n_steps = 1000
            x = np.linspace(-domain_width/2, domain_width/2, n_steps)
            
            # Combine multiple mathematical patterns
            hybrid_func = np.zeros(n_steps)
            
            # Pattern 1: Smooth cosine-based envelope
            hybrid_func += 10 * np.cos(1.0 * np.pi * x / (domain_width/2))
            
            # Pattern 2: Gaussian-like pulses with specific spacing
            for i in range(5):
                center = -domain_width/2 + (i + 1) * (domain_width/6)
                width = 0.05
                amplitude = 15 * np.exp(-0.5 * ((x - center) / width)**2)
                hybrid_func += amplitude
            
            # Pattern 3: Additional modulation
            hybrid_func += 5 * np.sin(2.0 * np.pi * x / (domain_width/2))
            
            # Apply constraints and normalization
            hybrid_func = np.maximum(hybrid_func, 0)
            if np.max(hybrid_func) > 0:
                hybrid_func = hybrid_func / np.max(hybrid_func) * 30
            
            hybrid_list = hybrid_func.tolist()
            c2_hybrid, _ = evaluate_function(hybrid_list)
            
            if c2_hybrid > best_c2:
                best_c2 = c2_hybrid
                best_f = hybrid_list.copy()
        except Exception:
            pass
    
    # Final fallback to a robust baseline if needed
    if best_f is None:
        best_f = [10.0] * 500
        best_c2 = evaluate_function(best_f)[0]
    
    # Convert to list format
    result = best_f
    
    # Post-processing to ensure non-negativity
    result = [max(0, x) for x in result]
    
    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
