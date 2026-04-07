# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import warnings
from typing import List, Tuple
import random
import time

class SpectralPeakOptimizer:
    """
    Advanced spectral-guided step function optimizer with modular architecture
    """

    def __init__(self, max_time_seconds: int = 85):
        self.max_time_seconds = max_time_seconds
        self.start_time = None
        np.random.seed(42)
        random.seed(42)

    def _ensure_time_remaining(self, safety_margin: float = 5.0) -> bool:
        """Check if there's sufficient time remaining"""
        if self.start_time is None:
            self.start_time = time.time()
        return (time.time() - self.start_time) < (self.max_time_seconds - safety_margin)

    def _compute_autoconvolution_norms(self, f: List[float]) -> Tuple[float, float, float]:
        """
        Compute the three norms needed for C2 calculation using precise numerical integration
        Returns (||g||₂², ||g||₁, ||g||∞)
        """
        if not f:
            return 0.0, 0.0, 0.0
            
        f_arr = np.array(f, dtype=np.float64)
        
        # Compute autoconvolution using optimized convolution
        g = signal.convolve(f_arr, f_arr, mode='full')
        # Take only the central portion to avoid edge effects
        g = g[len(g)//2:]
        
        # Truncate to match original length
        if len(g) > len(f_arr):
            g = g[:len(f_arr)]
            
        # Compute norms with better numerical handling
        # L2 norm squared using proper integration weights
        norm_2_sq = 0.0
        dx = 0.5 / max(1, len(g) - 1)  # Step size
        
        if len(g) > 1:
            # Use trapezoidal integration for better accuracy
            for i in range(len(g) - 1):
                # Trapezoidal area: (dx/2) * (y_i + y_{i+1})^2  
                # But for L2 norm squared, we want sum of y_i^2 terms with integration weights
                # Using the standard approach: integral of f^2
                # We'll approximate with midpoint rule for L2 norm squared
                midpoint = (g[i] + g[i+1]) / 2.0
                norm_2_sq += midpoint * midpoint * dx
                
        # L1 norm: sum of absolute values times dx
        norm_1 = np.sum(np.abs(g)) * dx
        
        # L-infinity norm: maximum absolute value
        norm_inf = np.max(np.abs(g)) if len(g) > 0 else 0.0
        
        return norm_2_sq, norm_1, norm_inf

    def compute_c2(self, f: List[float]) -> float:
        """Compute C2 value with robust error handling"""
        if not f:
            return 0.0
            
        norm_2_sq, norm_1, norm_inf = self._compute_autoconvolution_norms(f)
        
        # Avoid division by zero or extremely small values
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
            
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2 if not np.isnan(c2) and not np.isinf(c2) else 0.0

    def _generate_spectral_guided_peaks(self, n_steps: int) -> List[Tuple[float, float, float]]:
        """Generate peaks using spectral-guided initialization strategy"""
        if not self._ensure_time_remaining():
            return []
            
        x = np.linspace(-0.25, 0.25, n_steps)
        peaks = []

        # Central dense region with strategic peaks
        central_count = np.random.randint(8, 16)
        for i in range(central_count):
            pos = np.random.uniform(-0.1, 0.1)
            height = np.random.uniform(1.8, 2.5)
            width = np.random.uniform(0.015, 0.035)
            peaks.append((pos, height, width))

        # Mid-region with moderate density
        mid_count = np.random.randint(6, 10)
        for i in range(mid_count):
            pos = np.random.choice([-0.2, -0.18, -0.16, -0.14, -0.12, -0.1,
                                   0.1, 0.12, 0.14, 0.16, 0.18, 0.2])
            height = np.random.uniform(1.5, 2.2)
            width = np.random.uniform(0.025, 0.045)
            peaks.append((pos, height, width))

        # Outer sparse region
        outer_count = np.random.randint(3, 7)
        for i in range(outer_count):
            pos = np.random.choice([-0.24, -0.22, -0.2, 0.2, 0.22, 0.24])
            height = np.random.uniform(1.2, 1.8)
            width = np.random.uniform(0.035, 0.06)
            peaks.append((pos, height, width))

        return peaks

    def _add_supplementary_structure(self, f_values: np.ndarray, x: np.ndarray):
        """Add supplementary structure to enhance autoconvolution properties"""
        if not self._ensure_time_remaining():
            return
            
        n_steps = len(f_values)
        for i in range(0, n_steps, max(1, n_steps//30)):
            if np.random.random() > 0.85:
                bump_center = x[i]
                bump_height = np.random.uniform(0.05, 0.2)
                bump_width = np.random.uniform(0.008, 0.018)
                bump = bump_height * np.exp(-0.5 * ((x - bump_center) / bump_width)**2)
                f_values += bump

    def _identify_key_peaks(self, func_vals: np.ndarray, x: np.ndarray, 
                          min_height_ratio: float = 0.1) -> List[float]:
        """Identify key peak locations in the function"""
        if not self._ensure_time_remaining():
            return []
            
        peaks = []
        for i in range(1, len(func_vals)-1):
            if (func_vals[i] > func_vals[i-1] and 
                func_vals[i] > func_vals[i+1] and
                func_vals[i] > min_height_ratio * np.max(func_vals)):
                peaks.append(x[i])
                
        # Fallback if no peaks detected
        if not peaks:
            return [x[len(x)//2], x[len(x)//4], x[3*len(x)//4]]
            
        return sorted(set(peaks))

    def _reconstruct_from_peaks(self, peak_params: List[float], x: np.ndarray) -> np.ndarray:
        """Reconstruct function from peak parameters"""
        if not self._ensure_time_remaining():
            return np.zeros_like(x)
            
        func = np.zeros_like(x)
        
        # Each group of 3 parameters represents a peak: [position, height, width]
        for i in range(0, len(peak_params), 3):
            if i + 2 < len(peak_params):
                pos = peak_params[i]
                height = peak_params[i+1]
                width = peak_params[i+2]
                
                gaussian_peak = height * np.exp(-0.5 * ((x - pos) / width)**2)
                func += gaussian_peak
                
        return func

    def _create_initial_function(self, n_steps: int) -> List[float]:
        """Create initial step function with spectral guidance"""
        if not self._ensure_time_remaining():
            return [1.0] * 100
            
        x = np.linspace(-0.25, 0.25, n_steps)
        base_function = np.zeros_like(x)
        
        # Generate spectral-guided peaks
        spectral_peaks = self._generate_spectral_guided_peaks(n_steps)
        
        # Apply peaks
        for pos, height, width in spectral_peaks:
            gaussian_peak = height * np.exp(-0.5 * ((x - pos) / width)**2)
            base_function += gaussian_peak
            
        # Add supplementary structure
        self._add_supplementary_structure(base_function, x)
        
        # Ensure non-negativity and normalize
        base_function = np.maximum(base_function, 0)
        if np.max(base_function) > 0:
            base_function = base_function / np.max(base_function) * 2.0
            
        # Apply smoothing
        window_size = min(51, max(3, n_steps // 100))
        if window_size % 2 == 0:
            window_size += 1
        if window_size > 1:
            window = np.ones(window_size) / window_size
            base_function = np.convolve(base_function, window, mode='same')
            
        return base_function.tolist()

    def _evolve_peak_parameters(self, initial_func: List[float], n_steps: int) -> List[float]:
        """Selective evolutionary optimization of peak parameters"""
        if not self._ensure_time_remaining():
            return initial_func
            
        x = np.linspace(-0.25, 0.25, n_steps)
        current_func = np.array(initial_func)
        
        # Identify key peaks
        peak_locations = self._identify_key_peaks(current_func, x)
        
        # Create peak parameter vector: [pos1, height1, width1, pos2, height2, width2...]
        peak_params = []
        for loc in peak_locations:
            idx = np.argmin(np.abs(x - loc))
            height = current_func[idx]
            width = 0.025  # Default width
            peak_params.extend([loc, height, width])
            
        if not peak_params:
            return initial_func
            
        # Initialize variables
        best_params = peak_params.copy()
        best_c2 = self.compute_c2(current_func)
        
        # Evolutionary optimization setup
        population_size = 20
        max_generations = 30
        mutation_rate = 0.1
        
        # Initialize population
        population = [best_params.copy()]
        for _ in range(population_size - 1):
            individual = [p + random.gauss(0, 0.05 * abs(p) if p != 0 else 0.05)
                         for p in best_params]
            population.append(individual)
            
        # Evolution loop
        for generation in range(max_generations):
            if not self._ensure_time_remaining():
                break
                
            # Evaluate population
            fitness_scores = []
            evaluated_individuals = []
            
            for individual in population:
                reconstructed_func = self._reconstruct_from_peaks(individual, x)
                reconstructed_func = np.maximum(reconstructed_func, 0)
                
                c2 = self.compute_c2(reconstructed_func)
                fitness_scores.append(c2)
                evaluated_individuals.append((individual, c2))
                
            # Update best
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_c2:
                best_c2 = fitness_scores[best_idx]
                best_params = evaluated_individuals[best_idx][0].copy()
                
            # Create new population
            new_population = []
            for _ in range(population_size):
                if not self._ensure_time_remaining():
                    break
                    
                # Tournament selection
                tournament_size = 3
                tournament_indices = random.choices(range(population_size), k=tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                
                # Clone and mutate
                new_individual = population[winner_idx].copy()
                for i in range(len(new_individual)):
                    if random.random() < mutation_rate:
                        mutation_strength = 0.1 * abs(new_individual[i]) if new_individual[i] != 0 else 0.1
                        new_individual[i] += random.gauss(0, mutation_strength)
                        
                new_population.append(new_individual)
                
            population = new_population
            
        # Return best result
        best_function = self._reconstruct_from_peaks(best_params, x)
        return best_function.tolist()

    def construct_function(self) -> List[float]:
        """
        Main function to construct step-function with high C2 value
        """
        # Use fixed number of steps to match benchmark requirements
        n_steps = 5000
        
        # Create initial function
        try:
            initial_func = self._create_initial_function(n_steps)
            
            # Apply evolutionary refinement if time permits
            if self._ensure_time_remaining():
                refined_func = self._evolve_peak_parameters(initial_func, n_steps)
                final_func = refined_func
            else:
                final_func = initial_func
                
        except Exception as e:
            warnings.warn(f"Construction failed: {str(e)}")
            # Fallback to simple construction
            final_func = [1.0] * n_steps
            
        # Add final robustness
        final_array = np.array(final_func)
        noise = np.random.normal(0, 0.005, len(final_array))
        final_array += noise
        final_array = np.maximum(final_array, 0)
        
        return final_array.tolist()

def construct_function() -> list[float]:
    """Interface function that calls the spectral peak optimizer"""
    optimizer = SpectralPeakOptimizer(max_time_seconds=85)
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")