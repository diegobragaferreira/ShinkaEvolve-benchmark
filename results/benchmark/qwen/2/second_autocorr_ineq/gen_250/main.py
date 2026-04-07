# EVOLVE-BLOCK-START

import numpy as np
from typing import List, Tuple
import time
import random
from scipy import signal
from functools import lru_cache
from numba import jit, prange
import numba

class OptimizedStepFunctionOptimizer:
    """High-performance optimizer for step function construction to maximize C2."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Configuration parameters
        self.domain_width = 0.5
        self.domain_center = 0.0
        self.default_step_count = 5000  # Match AlphaEvolve benchmark
        
    @staticmethod
    @jit(nopython=True, cache=True)
    def compute_autoconvolution_numba(f_values: list) -> np.ndarray:
        """Numba-accelerated autoconvolution computation"""
        n = len(f_values)
        if n == 0:
            return np.array([])
        
        g = np.zeros(2*n - 1, dtype=np.float64)
        
        # Manual convolution loop for speed - fully vectorized Numba
        for i in range(n):
            for j in range(n):
                g[i + j] += f_values[i] * f_values[j]
        
        # Return positive lags only - this avoids data copying
        return g[n-1:] 
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def compute_norms_numba(g_values: np.ndarray) -> Tuple[float, float, float]:
        """Numba-accelerated norm computations with precise integration"""
        n = len(g_values)
        if n == 0:
            return 0.0, 0.0, 0.0
        
        # Compute norms using optimized Numba loops
        norm_1 = 0.0
        norm_2_sq = 0.0
        norm_inf = 0.0
        
        # L1 norm computation
        for i in range(n):
            abs_g = abs(g_values[i])
            norm_1 += abs_g
        
        # L2 norm squared using precise piecewise integration
        # Using cubic integration formula (dx/3)(y0^2 + y0*y1 + y1^2) 
        norm_2_sq = 0.0
        if n > 1:
            dx = 1.0  # Normalized spacing
            for i in range(n-1):
                y0 = g_values[i]
                y1 = g_values[i+1] 
                norm_2_sq += (dx/3.0) * (y0*y0 + y0*y1 + y1*y1)
        
        # Infinity norm
        for i in range(n):
            abs_g = abs(g_values[i])
            if abs_g > norm_inf:
                norm_inf = abs_g
        
        return norm_1, norm_2_sq, norm_inf
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def compute_c2_numba(norm_1: float, norm_2_sq: float, norm_inf: float) -> float:
        """Numba-accelerated C2 computation with safe division"""
        if norm_1 < 1e-12 or norm_inf < 1e-12:
            return 0.0
        return norm_2_sq / (norm_1 * norm_inf)
    
    def compute_c2(self, f_values: List[float]) -> float:
        """Fast C2 computation using Numba-optimized functions"""
        try:
            # Fast autoconvolution
            g = self.compute_autoconvolution_numba(f_values)
            
            # Fast norm computations
            norm_1, norm_2_sq, norm_inf = self.compute_norms_numba(g)
            
            # C2 computation
            c2 = self.compute_c2_numba(norm_1, norm_2_sq, norm_inf)
            
            return c2
        except Exception:
            return 0.0
    
    def gaussian_peak_function(self, x: np.ndarray, peak_params: List[float]) -> np.ndarray:
        """Generate function from Gaussian peak parameters."""
        result = np.zeros_like(x)
        for i in range(0, len(peak_params), 3):
            amp, center, width = peak_params[i], peak_params[i+1], peak_params[i+2]
            width = max(width, 1e-6)
            result += amp * np.exp(-0.5 * ((x - center) / width)**2)
        return result
    
    def enforce_peak_spacing_optimized(self, peak_params: List[float], 
                                     min_distance_ratio: float = 0.05) -> None:
        """Efficient peak spacing enforcement using sorted processing."""
        if len(peak_params) < 3:
            return
        
        # Create sorted list of peaks by center position
        peaks = []
        for i in range(0, len(peak_params), 3):
            peaks.append([peak_params[i], peak_params[i+1], peak_params[i+2]])
        
        # Sort by center position (stable sort)
        peaks.sort(key=lambda x: x[1])
        
        # Process pairs to enforce minimum spacing
        min_distance = min_distance_ratio * self.domain_width
        i = 1
        while i < len(peaks):
            prev_center = peaks[i-1][1]
            curr_center = peaks[i][1]
            distance = abs(curr_center - prev_center)
            
            if distance < min_distance:
                # Adjust position of current peak to maintain spacing
                offset = min_distance - distance
                if curr_center > prev_center:
                    peaks[i][1] += offset
                else:
                    peaks[i][1] -= offset
                # Re-sort to maintain order
                peaks.sort(key=lambda x: x[1])
                # Reset index to recheck from beginning
                i = 1
            else:
                i += 1
        
        # Put peaks back into flat list
        for i, (amp, center, width) in enumerate(peaks):
            peak_params[i*3] = amp
            peak_params[i*3 + 1] = center
            peak_params[i*3 + 2] = width
    
    def create_individual(self, num_peaks: int) -> List[float]:
        """Create a random individual with specified number of Gaussian peaks"""
        individual = []
        for _ in range(num_peaks):
            # Amplitude: 10 to 100 (log-uniform for better spread)
            individual.append(10**(np.random.uniform(1.0, 2.0)))
            # Center: -domain_width/2 to domain_width/2 (log-uniform distribution for better spread)
            individual.append(np.random.uniform(-self.domain_width/2, self.domain_width/2))
            # Width: 0.01 to 0.2 (log-uniform)
            individual.append(0.01 * (10**(np.random.uniform(0.0, 1.0))))
        return individual
    
    def evaluate_individual(self, individual: List[float]) -> float:
        """Fast evaluation of individual fitness"""
        try:
            # Generate function from peak parameters
            domain_points = np.linspace(-self.domain_width/2, self.domain_width/2, self.default_step_count)
            func_values = self.gaussian_peak_function(domain_points, individual)
            # Convert to step function values
            step_values = func_values.tolist()
            # Compute C2
            c2_value = self.compute_c2(step_values)
            return c2_value
        except Exception:
            return 0.0
    
    def optimize_with_evolutionary_algorithm(self, num_peaks: int) -> List[float]:
        """Optimized evolutionary algorithm with selective mutation"""
        # Start with best initial solution
        best_c2 = 0.0
        best_individual = self.create_individual(num_peaks)
        
        # Run multiple rounds of optimization with different strategies
        for round_num in range(3):
            current_individual = list(best_individual)
            
            # Different strategies for each round
            if round_num == 0:
                # Aggressive mutation for wide exploration
                max_mutations = 200
                mutation_rate = 0.5
            elif round_num == 1:
                # Moderate mutation for fine-tuning 
                max_mutations = 150
                mutation_rate = 0.3
            else:
                # Conservative refinement
                max_mutations = 100
                mutation_rate = 0.1
            
            for iteration in range(max_mutations):
                mutated_individual = list(current_individual)
                
                # Select parameters to mutate using probabilistic approach
                params_to_mutate = []
                for i in range(len(mutated_individual)):
                    if np.random.random() < mutation_rate:
                        params_to_mutate.append(i)
                
                # Apply mutations selectively
                for param_index in params_to_mutate:
                    if param_index % 3 == 0:  # amplitude - log-uniform mutation
                        log_amp = np.log10(mutated_individual[param_index])
                        log_amp += np.random.uniform(-0.3, 0.3)
                        mutated_individual[param_index] = max(0.1, 10 ** log_amp)
                    elif param_index % 3 == 1:  # center - linear
                        mutated_individual[param_index] += np.random.uniform(-0.02, 0.02)
                        # Keep in bounds
                        mutated_individual[param_index] = max(-self.domain_width/2, 
                                                            min(self.domain_width/2, 
                                                                mutated_individual[param_index]))
                    else:  # width - log-uniform
                        log_width = np.log10(mutated_individual[param_index])
                        log_width += np.random.uniform(-0.2, 0.2)
                        mutated_individual[param_index] = max(0.001, 10 ** log_width)
                
                # Ensure non-negativity and enforce peak spacing
                for i in range(len(mutated_individual)):
                    mutated_individual[i] = max(0, mutated_individual[i])
                
                self.enforce_peak_spacing_optimized(mutated_individual)
                
                # Evaluate
                c2_value = self.evaluate_individual(mutated_individual)
                
                if c2_value > best_c2:
                    best_c2 = c2_value
                    best_individual = list(mutated_individual)
                    
                    # Early termination based on improvement
                    if round_num == 0 and best_c2 > 0.95:
                        break
        
        return best_individual
    
    def adaptive_refinement(self, f_values: List[float], max_iterations: int = 500) -> List[float]:
        """Advanced refinement with adaptive step sizing and simulated annealing"""
        current_f = list(f_values)
        current_c2 = self.compute_c2(current_f)
        
        improvement_count = 0
        step_size = 0.1
        prev_c2 = current_c2
        temperature = 1.0
        cooling_rate = 0.995
        
        for iteration in range(max_iterations):
            modified_f = list(current_f)
            idx = np.random.randint(len(modified_f))
            
            # Adaptive perturbation with temperature
            delta = np.random.normal(0, step_size * temperature)
            modified_f[idx] = max(0.0, modified_f[idx] + delta)
            
            test_c2 = self.compute_c2(modified_f)
            
            # Accept improvement or with probability based on temperature
            if test_c2 > current_c2 or np.random.random() < np.exp((test_c2 - current_c2) / (temperature + 1e-8)):
                current_f = modified_f
                current_c2 = test_c2
                improvement_count = 0
            else:
                improvement_count += 1
                
            # Adjust step size and temperature
            if improvement_count > 5:
                step_size *= 0.9
                improvement_count = 0
                
            temperature *= cooling_rate  # Cool down
            
            # Early stopping
            if improvement_count > 20:
                break
                
            # Progress monitoring
            if abs(test_c2 - prev_c2) < 1e-8:
                improvement_count += 1
            else:
                improvement_count = 0
                
            prev_c2 = current_c2
            
        return current_f
    
    def construct_function(self) -> List[float]:
        """Main function to construct step-function with high C2 value."""
        start_time = time.time()
        best_c2 = 0.0
        best_function = []
        
        # Strategy 1: Multi-start with evolutionary algorithm using different peak counts
        peak_counts = [3, 5, 7, 10, 15]  # Reduced range for faster processing
        
        for num_peaks in peak_counts:
            if time.time() - start_time > 80:  # Leave buffer for final refinement
                break
                
            try:
                # Optimized evolutionary optimization
                peak_params = self.optimize_with_evolutionary_algorithm(num_peaks)
                
                # Generate function
                domain_points = np.linspace(-self.domain_width/2, self.domain_width/2, self.default_step_count)
                func_values = self.gaussian_peak_function(domain_points, peak_params)
                step_values = func_values.tolist()
                c2_val = self.compute_c2(step_values)
                
                if c2_val > best_c2:
                    best_c2 = c2_val
                    best_function = step_values
                    
            except Exception as e:
                continue
        
        # Strategy 2: Final refinement around best solution if available
        if len(best_function) > 0 and time.time() - start_time < 85:
            try:
                refined_f = self.adaptive_refinement(best_function, max_iterations=200)
                final_c2 = self.compute_c2(refined_f)
                
                if final_c2 > best_c2:
                    best_c2 = final_c2
                    best_function = refined_f
            except Exception as e:
                pass
        
        # Strategy 3: Quick fallback to Gaussian-based function if needed
        if len(best_function) == 0 and time.time() - start_time < 85:
            try:
                # Create a reasonably shaped Gaussian function
                x = np.linspace(-0.25, 0.25, self.default_step_count)
                # Use a single broad Gaussian as fallback
                g = 50 * np.exp(-0.5 * (x/0.15)**2)
                best_function = g.tolist()
                best_c2 = self.compute_c2(best_function)
            except Exception as e:
                pass
        
        # Strategy 4: Final fallback to uniform distribution
        if len(best_function) == 0:
            best_function = [10.0] * self.default_step_count
            
        return best_function

def construct_function() -> List[float]:
    """Main entry point function for constructing step function with high C2 value."""
    optimizer = OptimizedStepFunctionOptimizer()
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
