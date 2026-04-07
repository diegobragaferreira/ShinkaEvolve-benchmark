# EVOLVE-BLOCK-START

import numpy as np
from numba import njit
import random
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

@njit
def compute_autoconvolution_norms_fast(f_values):
    """
    Fast FFT-based autoconvolution computation with improved integration.
    """
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Use FFT-based convolution for performance
    padded_length = 2 * n - 1
    padded_f = np.pad(f_values, (0, padded_length - n), mode='constant')

    # FFT convolution: f * f = ifft(fft(f) * fft(f))
    fft_f = np.fft.fft(padded_f)
    conv_fft = fft_f * fft_f
    g = np.fft.ifft(conv_fft).real

    # Take only the linear portion
    g = g[:padded_length]

    # Compute norms using more accurate integration
    l2_norm_squared = 0.0
    if len(g) >= 2:
        # For L2 norm squared, integrate g² with trapezoidal rule
        for i in range(len(g) - 1):
            y1 = g[i]
            y2 = g[i+1]
            # Using (h/3)(y1² + y1*y2 + y2²) for piecewise quadratic integration
            l2_norm_squared += (y1*y1 + y1*y2 + y2*y2) / 3.0

    # L1 norm
    l1_norm = np.sum(np.abs(g)) / (len(g) + 1)

    # L-infinity norm
    l_inf_norm = np.max(np.abs(g))

    return l2_norm_squared, l1_norm, l_inf_norm

@njit
def calculate_c2_fast(l2_norm_squared, l1_norm, l_inf_norm):
    """Fast C2 calculation with numerical stability"""
    if l1_norm <= 1e-15 or l_inf_norm <= 1e-15:
        return 0.0
    return l2_norm_squared / (l1_norm * l_inf_norm)

def compute_autoconvolution_norms(f_values):
    """Wrapper that ensures robust computation"""
    try:
        return compute_autoconvolution_norms_fast(f_values)
    except:
        return 0.0, 0.0, 0.0

def calculate_c2(f_values):
    """Calculate C₂ from step function values"""
    try:
        norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
        return calculate_c2_fast(norm_2_squared, norm_1, norm_inf)
    except:
        return 0.0

def generate_geometric_pattern(length):
    """Generate structured geometric pattern designed to produce good C2"""
    # Create multi-scale structure using Gaussian bumps with different scales
    individual = np.zeros(length)

    # Add several multi-scale bumps for complex yet structured behavior
    num_bumps = min(15, length // 8)

    for _ in range(num_bumps):
        # Position and scale parameters
        center = np.random.randint(0, length)
        width = np.random.randint(1, max(3, length // 15))
        height = np.random.uniform(0.6, 1.4)

        # Fill Gaussian shape with proper normalization
        for i in range(length):
            distance = abs(i - center)
            if distance < width * 4:  # Only compute relevant region
                gaussian_val = height * np.exp(-0.5 * (distance / width)**2)
                individual[i] += gaussian_val

    # Add some structured noise to avoid over-fitting to specific patterns
    for i in range(length):
        if np.random.random() < 0.3:
            individual[i] = max(0.0, individual[i] + np.random.normal(0, 0.15))

    return individual.tolist()

def generate_harmonic_pattern(length):
    """Create a harmonic pattern that balances frequency components"""
    individual = []

    # Create pattern using sine/cosine combinations to introduce regularity
    t = np.linspace(0, 4*np.pi, length)

    # Base component
    base = 0.5 + 0.3 * np.sin(t)

    # Add harmonic components
    harmonics = 0.2 * np.sin(2*t) + 0.1 * np.cos(3*t) + 0.1 * np.sin(4*t)

    # Combine and add some randomization
    combined = base + harmonics

    # Convert to list and ensure non-negativity
    individual = [max(0.0, val) for val in combined]

    # Add structured noise
    for i in range(len(individual)):
        if np.random.random() < 0.2:
            individual[i] = max(0.0, individual[i] + np.random.normal(0, 0.1))

    return individual

def generate_combination_pattern(length):
    """Generate hybrid pattern combining geometric and harmonic structures"""
    # Generate both types of patterns
    geo_pattern = generate_geometric_pattern(length)
    harm_pattern = generate_harmonic_pattern(length)

    # Combine them with weighted average
    combined = []
    for i in range(length):
        # Blend with preference for geometric pattern
        blended = 0.6 * geo_pattern[i] + 0.4 * harm_pattern[i]
        combined.append(blended)

    # Add some structured randomness
    for i in range(len(combined)):
        if np.random.random() < 0.15:
            combined[i] = max(0.0, combined[i] + np.random.normal(0, 0.1))

    return combined

def generate_spectral_pattern(length):
    """Generate pattern based on spectral analysis for good C2 performance"""
    # Create a pattern that has both low and high frequency components
    t = np.linspace(0, 4*np.pi, length)
    
    # Base with multiple frequencies
    base = 0.5 + 0.3 * np.sin(t) + 0.2 * np.cos(2*t) + 0.1 * np.sin(3*t)
    
    # Add some structured noise 
    noise = np.random.normal(0, 0.1, length)
    
    # Combine and ensure non-negativity
    pattern = base + noise
    pattern = np.maximum(pattern, 0.0)
    
    return pattern.tolist()

def generate_alternating_pattern(length):
    """Generate alternating high/medium/low pattern for structure diversity"""
    pattern = []
    for i in range(length):
        # Cycle through pattern types
        cycle = i % 6
        if cycle < 2:
            pattern.append(np.random.uniform(0.8, 1.2))  # High
        elif cycle < 4:
            pattern.append(np.random.uniform(0.4, 0.8))  # Medium
        else:
            pattern.append(np.random.uniform(0.1, 0.4))  # Low
    
    # Add some smoothing
    smoothed = []
    for i in range(length):
        avg_val = pattern[i]
        if i > 0:
            avg_val = 0.7 * avg_val + 0.3 * pattern[i-1]
        if i < length - 1:
            avg_val = 0.7 * avg_val + 0.3 * pattern[i+1]
        smoothed.append(avg_val)
    
    return smoothed

def adaptive_local_refinement(seed_solution, iterations=10):
    """Perform adaptive refinement around a good solution"""
    current = np.array(seed_solution)

    for _ in range(iterations):
        # Try small random perturbations
        candidate = current.copy()

        # Apply perturbations to random subset of elements
        num_perturb = max(1, len(candidate) // 20)  # Perturb ~5% of elements
        indices = np.random.choice(len(candidate), num_perturb, replace=False)

        for i in indices:
            # Adaptive perturbation based on current value
            noise_scale = max(0.05, 0.02 * candidate[i]) if candidate[i] > 0 else 0.05
            candidate[i] = max(0.0, candidate[i] + np.random.normal(0, noise_scale))

        # Test improvement
        try:
            current_c2 = calculate_c2(current.tolist())
            candidate_c2 = calculate_c2(candidate.tolist())

            if candidate_c2 > current_c2:
                current = candidate
        except:
            continue  # Skip failed attempts

    return current.tolist()

def simple_direct_optimization(initial_solution, max_iterations=100):
    """Use direct Nelder-Mead optimization on a single starting point"""
    # Wrap the objective function for scipy optimization
    def objective(x):
        try:
            # Clip negative values to prevent errors
            x_clipped = np.maximum(x, 0.0)
            # Return negative since we want to maximize
            return -calculate_c2(x_clipped.tolist())
        except:
            return 1e10  # Penalty for invalid solutions
    
    # Optimize using Nelder-Mead
    try:
        result = minimize(
            objective, 
            initial_solution, 
            method='Nelder-Mead',
            options={'maxiter': max_iterations, 'xtol': 1e-6, 'ftol': 1e-6}
        )
        
        if result.success:
            final_solution = np.maximum(result.x, 0.0)
            final_c2 = calculate_c2(final_solution.tolist())
            return final_solution.tolist(), final_c2
    except:
        pass
    
    # Return original if optimization fails
    final_c2 = calculate_c2(initial_solution)
    return initial_solution, final_c2

def multi_restart_optimization():
    """Run multiple optimizations with different starting points"""
    best_c2 = 0.0
    best_solution = None
    
    # Various initialization strategies
    strategies = [
        lambda n: generate_geometric_pattern(n),
        lambda n: generate_harmonic_pattern(n),
        lambda n: generate_combination_pattern(n),
        lambda n: generate_spectral_pattern(n),
        lambda n: generate_alternating_pattern(n)
    ]
    
    # Different restarts
    total_restarts = 15
    successful_restarts = 0
    
    for restart in range(total_restarts):
        try:
            # Randomly choose a strategy and length
            strategy_idx = restart % len(strategies)
            strategy = strategies[strategy_idx]
            length = np.random.randint(500, 1500)
            
            # Generate initial solution
            initial_solution = strategy(length)
            
            # Apply adaptive refinement on start
            refined_start = adaptive_local_refinement(initial_solution, 3)
            
            # Use direct optimization
            final_solution, final_c2 = simple_direct_optimization(refined_start, 50)
            
            if final_c2 > best_c2:
                best_c2 = final_c2
                best_solution = final_solution
            
            successful_restarts += 1
            
        except Exception as e:
            continue  # Skip failed restarts
    
    # Final local refinement on the best solution found
    if best_solution is not None and best_c2 > 0:
        try:
            final_solution = adaptive_local_refinement(best_solution, 10)
            final_c2 = calculate_c2(final_solution)
            if final_c2 > best_c2:
                best_c2 = final_c2
                best_solution = final_solution
        except:
            pass  # Continue with previous best
    
    return best_solution, best_c2

def construct_function() -> list[float]:
    """Main function that constructs step-function with high C2 value."""
    start_time = time.time()

    # Multi-restart direct optimization approach
    best_solution, best_c2 = multi_restart_optimization()

    elapsed = time.time() - start_time
    if elapsed > 85:
        # Return heuristic if time limit is near
        return [np.random.random() for _ in range(500)]

    # Return final best solution or default
    return best_solution if best_solution is not None else [0.5] * 1000

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")