# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
from scipy import signal
import math

@jit(nopython=True)
def compute_convolution_norms_numba(f_values, domain_length=0.5):
    """
    Fast computation of convolution norms using numba optimization
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step size
    dx = domain_length / n_steps

    # Compute autoconvolution g = f * f using direct computation
    g_size = 2 * n_steps - 1
    g = np.zeros(g_size)

    # Direct convolution computation
    for i in range(n_steps):
        for j in range(n_steps):
            k = i + j
            if 0 <= k < g_size:
                g[k] += f_values[i] * f_values[j] * dx

    # Compute norms using piecewise linear integration approach
    # For ||g||₂² using trapezoidal-like formula: (dx/3)(g₀² + g₀g₁ + g₁²)
    g2_sq = 0.0
    for i in range(len(g)-1):
        g2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)

    # ||g||₁ = sum(|g_i| * dx)
    g1 = np.sum(np.abs(g)) * dx

    # ||g||∞ = max(|g_i|)
    ginf = np.max(np.abs(g))

    return g2_sq, g1, ginf

def compute_c2(f_values):
    """Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞)"""
    g2_sq, g1, ginf = compute_convolution_norms_numba(f_values)

    if g1 == 0 or ginf == 0:
        return 0.0

    return g2_sq / (g1 * ginf)

def create_harmonic_basis(n_steps, max_freq=10):
    """
    Create basis functions for harmonic decomposition
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    basis = []
    
    # Add cosine and sine basis functions
    for freq in range(1, max_freq + 1):
        # Cosine basis
        basis.append(np.cos(2 * np.pi * freq * x))
        # Sine basis  
        basis.append(np.sin(2 * np.pi * freq * x))
    
    return np.array(basis)

def harmonic_step_function(n_steps, amplitude_spectrum):
    """
    Construct step function using harmonic synthesis with Poisson kernel properties
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Initialize with Poisson kernel structure
    # This helps create functions with good convolution properties
    poisson_kernel = np.exp(-np.abs(x) / 0.1)
    poisson_kernel = poisson_kernel / np.max(poisson_kernel)  # Normalize
    
    # Add harmonic components based on amplitude spectrum
    step_func = poisson_kernel.copy()
    
    # Add multiple frequency components
    for i, amp in enumerate(amplitude_spectrum):
        if i % 2 == 0:
            # Cosine component
            freq = (i // 2) + 1
            step_func += amp * np.cos(2 * np.pi * freq * x)
        else:
            # Sine component
            freq = ((i - 1) // 2) + 1
            step_func += amp * np.sin(2 * np.pi * freq * x)
    
    # Ensure non-negativity
    step_func = np.maximum(step_func, 0)
    
    # Normalize to have reasonable magnitude
    if np.sum(step_func) > 0:
        step_func = step_func / np.sum(step_func) * 10
    
    return step_func.tolist()

def create_optimal_harmonic_pattern(n_steps):
    """
    Create an optimized harmonic pattern based on theoretical analysis
    """
    # Based on mathematical analysis of optimal convolution profiles
    # We want to avoid sharp peaks in the convolution
    # Use a combination of low frequencies for smoothness 
    # and some higher frequencies for complexity
    
    # Create amplitude spectrum with specific pattern
    # Low frequencies dominate to maintain smoothness
    # Small contributions from higher frequencies for complexity
    amplitude_spectrum = []
    
    # Add several harmonics with decreasing amplitudes
    for i in range(20):  # 20 harmonic components
        freq = i + 1
        # Amplitude decreases with frequency (higher frequencies contribute less)
        # but with a factor to preserve meaningful structure
        amp = 1.0 / (freq * (1.0 + 0.1 * freq))
        amplitude_spectrum.append(amp)
        
        # Add a few higher frequency components to add interesting structure
        if i < 5 and freq > 5:
            # Boost some higher frequencies slightly
            amplitude_spectrum[-1] *= 1.5
    
    # Use Poisson kernel as base shape
    x = np.linspace(-0.25, 0.25, n_steps)
    base_shape = np.exp(-np.abs(x) / 0.1)
    base_shape = base_shape / np.max(base_shape)
    
    # Combine with harmonic components
    step_func = base_shape.copy()
    
    for i, amp in enumerate(amplitude_spectrum[:10]):  # Use first 10 components
        if i % 2 == 0:
            freq = (i // 2) + 1
            step_func += amp * np.cos(2 * np.pi * freq * x)
        else:
            freq = ((i - 1) // 2) + 1
            step_func += amp * np.sin(2 * np.pi * freq * x)
    
    # Ensure non-negativity and normalize
    step_func = np.maximum(step_func, 0)
    if np.sum(step_func) > 0:
        step_func = step_func / np.sum(step_func) * 10
    
    return step_func.tolist()

def optimize_harmonic_functions():
    """
    Optimize step functions using harmonic approach with multi-start strategy
    """
    best_c2 = -np.inf
    best_function = None
    
    n_steps_list = [200, 300, 400, 500]
    
    # Multi-start with different harmonic configurations
    for n_steps in n_steps_list:
        # Try multiple harmonic setups
        for trial in range(5):
            np.random.seed(trial * 1000 + n_steps)
            
            # Generate different harmonic patterns
            if trial == 0:
                # Standard harmonic pattern
                f_values = create_optimal_harmonic_pattern(n_steps)
            elif trial == 1:
                # More complex harmonic pattern
                f_values = create_optimal_harmonic_pattern(n_steps)
                # Add some randomness
                for i in range(len(f_values)):
                    if np.random.random() < 0.2:
                        f_values[i] *= (1 + np.random.normal(0, 0.05))
            elif trial == 2:
                # Symmetric pattern
                x = np.linspace(-0.25, 0.25, n_steps)
                f_values = np.exp(-0.5 * (x / 0.1)**2)
                f_values = f_values.tolist()
            elif trial == 3:
                # Exponential decay pattern
                x = np.linspace(-0.25, 0.25, n_steps)
                f_values = np.exp(-np.abs(x) / 0.05)
                f_values = f_values.tolist()
            else:
                # Random harmonic pattern
                amplitude_spectrum = []
                for i in range(15):
                    amp = np.random.exponential(0.5) * (1.0 / (i + 1))
                    amplitude_spectrum.append(amp)
                f_values = harmonic_step_function(n_steps, amplitude_spectrum)
            
            # Ensure non-negativity
            f_values = [max(0, x) for x in f_values]
            
            # Normalize for better numerical behavior
            total = sum(f_values)
            if total > 0:
                f_values = [x / total * 10 for x in f_values]
            
            # Local refinement
            current_f = f_values.copy()
            current_c2 = compute_c2(current_f)
            
            # Gradient-like local search
            for iter in range(20):
                test_f = current_f.copy()
                # Modify a few points randomly
                indices = np.random.choice(len(test_f), min(10, len(test_f)//5), replace=False)
                for idx in indices:
                    # Small perturbation
                    change = np.random.normal(0, 0.05 * test_f[idx])
                    test_f[idx] = max(0, test_f[idx] + change)
                
                test_c2 = compute_c2(test_f)
                if test_c2 > current_c2:
                    current_c2 = test_c2
                    current_f = test_f
            
            # Check if this is our best solution
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_function = current_f.copy()
    
    return best_function

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using harmonic approach."""
    
    # Try the harmonic optimization approach
    try:
        solution = optimize_harmonic_functions()
        
        # If we got a solution, return it
        if solution is not None and len(solution) > 0:
            return solution
            
    except Exception as e:
        pass
    
    # Fallback: use simpler pattern-based approach
    n_steps = 300
    f_values = create_optimal_harmonic_pattern(n_steps)
    
    # Final refinement with some optimization
    current_f = f_values.copy()
    current_c2 = compute_c2(current_f)
    
    # Simple local search
    for _ in range(15):
        test_f = current_f.copy()
        # Make small random changes
        indices = np.random.choice(len(test_f), min(5, len(test_f)//4), replace=False)
        for idx in indices:
            change = np.random.normal(0, 0.1 * test_f[idx])
            test_f[idx] = max(0, test_f[idx] + change)
        
        test_c2 = compute_c2(test_f)
        if test_c2 > current_c2:
            current_c2 = test_c2
            current_f = test_f
    
    return current_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")