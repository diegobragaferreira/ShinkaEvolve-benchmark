# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
import time
import numba
from numba import jit
import random
from typing import List, Tuple
import math

# Constants
DOMAIN = [-0.25, 0.25]
N_MIN, N_MAX = 100, 2000
MAX_TIME_SECONDS = 85

@jit(nopython=True)
def compute_autoconvolution_fast(f_vals):
    """Fast numba-based autoconvolution computation"""
    n = len(f_vals)
    g = np.zeros(2*n - 1)
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]
    return g

@jit(nopython=True)
def compute_convolution_norms_fast(g_vals):
    """Fast computation of convolution norms"""
    n = len(g_vals)
    l2_sq = 0.0
    l1 = 0.0
    l_inf = 0.0

    for i in range(n):
        val = g_vals[i]
        l2_sq += val * val
        l1 += abs(val)
        if abs(val) > l_inf:
            l_inf = abs(val)

    return l2_sq, l1, l_inf

def compute_c2(f_vals):
    """Compute C2 value for given function values"""
    try:
        # Ensure non-negative values
        f_vals = np.maximum(f_vals, 0)
        
        # Compute autoconvolution
        g_vals = compute_autoconvolution_fast(f_vals)
        
        # Compute norms
        l2_sq, l1, l_inf = compute_convolution_norms_fast(g_vals)
        
        # Avoid division by zero
        if l1 <= 1e-12 or l_inf <= 1e-12:
            return 0.0
        
        # Compute C2
        c2 = l2_sq / (l1 * l_inf)
        return c2
    except Exception:
        return 0.0

class HarmonicPatternGenerator:
    """Generates step functions using harmonic patterns for better convolution behavior"""
    
    @staticmethod
    def generate_harmonic_initial_function(n: int, harmonics: int = 5) -> List[float]:
        """Generate initial function using additive harmonic components"""
        # Create base coordinate array
        x = np.linspace(-0.25, 0.25, n)
        
        # Generate base harmonic pattern with multiple frequencies
        pattern = np.zeros(n)
        
        # Add multiple sine/cosine components with different frequencies and amplitudes
        for i in range(1, harmonics + 1):
            freq = i * 2  # Frequencies: 2, 4, 6, ...
            amp = 1.0 / i  # Amplitude inversely proportional to frequency
            # Mix sine and cosine for richer patterns
            pattern += amp * (np.sin(freq * np.pi * x) + 0.5 * np.cos(freq * 2 * np.pi * x))
        
        # Apply sigmoid transformation to ensure non-negativity and smooth behavior
        pattern = 1.0 / (1.0 + np.exp(-5 * pattern)) * 2.0 - 1.0
        pattern = np.maximum(pattern, 0)
        
        # Normalize to reasonable magnitude
        total = np.sum(pattern)
        if total > 0:
            pattern = pattern / total * 5.0
            
        return pattern.tolist()
    
    @staticmethod
    def generate_multi_scale_harmonic(n: int) -> List[float]:
        """Generate multi-scale harmonic pattern that combines different resolution behaviors"""
        # Create pattern with multiple scales
        pattern = np.zeros(n)
        
        # Low frequency component for global structure
        x = np.linspace(0, 1, n)
        pattern += 0.5 * np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x)
        
        # Medium frequency component for intermediate detail
        pattern += 0.3 * np.sin(8 * np.pi * x) + 0.2 * np.cos(16 * np.pi * x)
        
        # High frequency component for fine detail
        pattern += 0.2 * np.sin(32 * np.pi * x) + 0.1 * np.cos(64 * np.pi * x)
        
        # Apply non-linear transformation to create desired shape
        pattern = np.abs(pattern)  # Make non-negative
        pattern = pattern ** 1.5  # Stretch distribution
        
        # Ensure non-negativity and normalize
        pattern = np.maximum(pattern, 0)
        total = np.sum(pattern)
        if total > 0:
            pattern = pattern / total * 10.0
            
        return pattern.tolist()

class AdaptiveHarmonicOptimizer:
    """Main optimizer using harmonic evolution approach"""
    
    def __init__(self):
        self.best_solution = None
        self.best_c2 = 0.0
    
    def adaptive_frequency_modulation(self, base_pattern: List[float], 
                                   generation: int, max_generations: int) -> List[float]:
        """Apply adaptive frequency modulation to improve solution quality"""
        pattern = np.array(base_pattern)
        n = len(pattern)
        
        # Adaptive frequency adjustment based on generation
        mod_freq_factor = 0.1 + 0.9 * (1.0 - generation / max_generations)
        
        # Add dynamic frequency modulation
        x = np.linspace(0, 1, n)
        modulation = np.sin(2 * np.pi * x * (1 + mod_freq_factor * generation))
        
        # Apply modulation with adaptive weights
        modified_pattern = pattern * (1 + 0.1 * modulation * mod_freq_factor)
        
        # Ensure non-negativity
        modified_pattern = np.maximum(modified_pattern, 0)
        
        # Normalize
        total = np.sum(modified_pattern)
        if total > 0:
            modified_pattern = modified_pattern / total * 10.0
            
        return modified_pattern.tolist()
    
    def convolution_aware_mutation(self, individual: List[float], 
                                 generation: int, max_generations: int) -> List[float]:
        """Apply mutations that consider convolution behavior"""
        mutated = np.array(individual)
        n = len(mutated)
        
        # Calculate current convolution to understand behavior
        current_c2 = compute_c2(individual)
        
        # Determine mutation intensity based on generation and current fitness
        mutation_intensity = 0.05 + 0.15 * (1.0 - generation / max_generations)
        
        # Apply convolution-aware perturbations
        for i in range(n):
            # If we're in early generations, do broader exploration
            if generation < max_generations * 0.3:
                # Larger random perturbations
                if random.random() < 0.1:
                    noise = np.random.normal(0, mutation_intensity * mutated[i])
                    mutated[i] = max(0, mutated[i] + noise)
            else:
                # More focused refinement
                if random.random() < 0.05:
                    noise = np.random.normal(0, mutation_intensity * mutated[i] * 0.5)
                    mutated[i] = max(0, mutated[i] + noise)
        
        # Normalize to maintain reasonable scale
        total = np.sum(mutated)
        if total > 0:
            mutated = mutated / total * 10.0
            
        return mutated.tolist()
    
    def multi_resolution_search(self, max_time_seconds: float) -> List[float]:
        """Multi-resolution search using harmonic patterns"""
        start_time = time.time()
        best_solution = None
        best_c2 = 0.0
        
        # Different resolution levels to explore
        resolutions = [100, 200, 300, 500, 700, 1000]
        
        for res in resolutions:
            if time.time() - start_time > max_time_seconds - 10:
                break
                
            # Generate initial harmonic pattern
            try:
                # Try different harmonic generation methods
                initial_patterns = [
                    HarmonicPatternGenerator.generate_harmonic_initial_function(res, 3),
                    HarmonicPatternGenerator.generate_harmonic_initial_function(res, 5),
                    HarmonicPatternGenerator.generate_multi_scale_harmonic(res)
                ]
                
                for pattern in initial_patterns:
                    if time.time() - start_time > max_time_seconds - 10:
                        break
                        
                    # Evaluate initial pattern
                    c2 = compute_c2(pattern)
                    
                    if c2 > best_c2:
                        best_c2 = c2
                        best_solution = pattern.copy()
                        
                    # Apply local refinement
                    refined = self.local_coordinate_refinement(pattern, max_time_seconds - (time.time() - start_time))
                    refined_c2 = compute_c2(refined)
                    
                    if refined_c2 > best_c2:
                        best_c2 = refined_c2
                        best_solution = refined.copy()
                        
            except Exception as e:
                continue
                
        return best_solution if best_solution is not None else [0.5] * 100
    
    def local_coordinate_refinement(self, initial_solution: List[float], 
                                  remaining_time: float) -> List[float]:
        """Refine solution with coordinate-wise local search"""
        current_solution = np.array(initial_solution)
        current_c2 = compute_c2(initial_solution)
        
        # Limited number of refinement iterations based on remaining time
        max_iterations = min(30, max(5, int(remaining_time * 10)))
        
        for iteration in range(max_iterations):
            improved = False
            # Try small perturbations to each element
            for i in range(len(current_solution)):
                original_val = current_solution[i]
                
                # Try different step sizes
                step_sizes = [0.01, 0.05, 0.1]
                for step in step_sizes:
                    for direction in [-1, 1]:
                        new_val = original_val + direction * step
                        if new_val >= 0:
                            test_solution = current_solution.copy()
                            test_solution[i] = new_val
                            new_c2 = compute_c2(test_solution.tolist())
                            
                            if new_c2 > current_c2:
                                current_c2 = new_c2
                                current_solution = test_solution
                                improved = True
                                break
                    if improved:
                        break
            if not improved:
                break
                
        return current_solution.tolist()

    def optimize(self, max_time_seconds: float) -> List[float]:
        """Main optimization routine using harmonic evolution approach"""
        start_time = time.time()
        
        # Multi-resolution search with harmonic patterns
        best_solution = self.multi_resolution_search(max_time_seconds)
        
        # Final refinement
        final_solution = self.local_coordinate_refinement(
            best_solution, max_time_seconds - (time.time() - start_time)
        )
        
        return final_solution

def construct_function() -> List[float]:
    """
    Function to construct step-function with high C2 value using harmonic evolution approach.
    """
    start_time = time.time()
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize optimizer
    optimizer = AdaptiveHarmonicOptimizer()
    
    # Execute the harmonic evolution optimization process
    result = optimizer.optimize(MAX_TIME_SECONDS)
    
    # Ensure we don't exceed time limit
    elapsed = time.time() - start_time
    if elapsed > MAX_TIME_SECONDS:
        return [0.5] * 100
        
    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")