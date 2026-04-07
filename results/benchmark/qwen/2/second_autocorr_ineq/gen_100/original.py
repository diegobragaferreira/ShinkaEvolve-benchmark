# EVOLVE-BLOCK-START

import numpy as np
from typing import List, Tuple, Optional
import time
import random
from scipy import signal
import math
from functools import lru_cache

class StepFunctionOptimizer:
    """Modular optimizer for step function construction to maximize C2."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        # Configuration parameters
        self.domain_width = 0.5
        self.domain_center = 0.0
        self.default_step_count = 500

    def compute_autoconvolution(self, f_values: List[float]) -> np.ndarray:
        """Compute the autoconvolution g = f * f of step function f."""
        n = len(f_values)
        if n == 0:
            return np.array([])

        f_array = np.array(f_values)
        g = np.convolve(f_array, f_array, mode='full')
        g = g[n-1:-(n-1)] if n > 1 else g
        return g

    def compute_norms(self, g_values: np.ndarray) -> Tuple[float, float, float]:
        """Compute the three required norms for C2 calculation."""
        if len(g_values) == 0:
            return 0.0, 0.0, 0.0

        # ||g||₂² using trapezoidal-like piecewise linear integration
        if len(g_values) <= 1:
            norm_2_sq = g_values[0]**2 if len(g_values) > 0 else 0.0
        else:
            norm_2_sq = 0.0
            for i in range(len(g_values)-1):
                h = 1.0
                norm_2_sq += (h/3.0) * (g_values[i]**2 + g_values[i]*g_values[i+1] + g_values[i+1]**2)

        # ||g||₁: L1 norm, approximated as sum(|g|) / (len(g) + 1)
        if len(g_values) > 0:
            norm_1 = np.sum(np.abs(g_values)) / (len(g_values) + 1)
        else:
            norm_1 = 0.0

        # ||g||∞: Infinity-norm
        norm_inf = np.max(np.abs(g_values)) if len(g_values) > 0 else 0.0

        return norm_2_sq, norm_1, norm_inf

    @lru_cache(maxsize=100)
    def cached_compute_c2(self, tuple_values: tuple) -> float:
        """Cached computation of C2 to avoid redundant calculations."""
        f_values = list(tuple_values)
        g = self.compute_autoconvolution(f_values)
        norm_2_sq, norm_1, norm_inf = self.compute_norms(g)

        if norm_1 == 0 or norm_inf == 0:
            return 0.0

        return norm_2_sq / (norm_1 * norm_inf)

    def compute_c2(self, f_values: List[float]) -> float:
        """Compute C2 for given step function values."""
        return self.cached_compute_c2(tuple(f_values))

    def gaussian_peak_function(self, x: np.ndarray, peak_params: List[float]) -> np.ndarray:
        """Generate a function composed of multiple Gaussian peaks."""
        result = np.zeros_like(x)
        for i in range(0, len(peak_params), 3):
            amp, center, width = peak_params[i], peak_params[i+1], peak_params[i+2]
            width = max(width, 1e-6)
            result += amp * np.exp(-0.5 * ((x - center) / width)**2)
        return result

    def enforce_peak_spacing(self, peak_params: List[float], min_distance_ratio: float = 0.05) -> None:
        """Enforce minimum distance between Gaussian peaks to prevent narrow autoconvolution."""
        if len(peak_params) < 3:
            return

        peaks = []
        for i in range(0, len(peak_params), 3):
            peaks.append([peak_params[i], peak_params[i+1], peak_params[i+2]])

        peaks.sort(key=lambda x: x[1])

        min_distance = min_distance_ratio * self.domain_width
        for i in range(1, len(peaks)):
            prev_center = peaks[i-1][1]
            curr_center = peaks[i][1]
            distance = abs(curr_center - prev_center)

            if distance < min_distance:
                offset = min_distance - distance
                if curr_center > prev_center:
                    peaks[i][1] += offset
                else:
                    peaks[i][1] -= offset

        for i, (amp, center, width) in enumerate(peaks):
            peak_params[i*3] = amp
            peak_params[i*3 + 1] = center
            peak_params[i*3 + 2] = width

    def create_individual(self, num_peaks: int) -> List[float]:
        """Create a random individual with specified number of Gaussian peaks"""
        individual = []
        for _ in range(num_peaks):
            # Amplitude: 10 to 100
            individual.append(random.uniform(10.0, 100.0))
            # Center: -domain_width/2 to domain_width/2
            individual.append(random.uniform(-self.domain_width/2, self.domain_width/2))
            # Width: 0.01 to 0.2
            individual.append(random.uniform(0.01, 0.2))
        return individual

    def evaluate_individual(self, individual: List[float]) -> float:
        """Evaluate the fitness of an individual (C2 value)"""
        try:
            # Generate function from peak parameters
            domain_points = np.linspace(-self.domain_width/2, self.domain_width/2, self.default_step_count)
            func_values = self.gaussian_peak_function(domain_points, individual)
            # Convert to step function values (take discrete samples)
            step_values = func_values.tolist()
            # Compute C2
            c2_value = self.compute_c2(step_values)
            return c2_value
        except Exception:
            return 0.0

    def optimize_with_evolutionary_algorithm(self, num_peaks: int,
                                           population_size: int = 30,
                                           generations: int = 30) -> List[float]:
        """Optimize peak parameters using evolutionary algorithm with enhanced control"""
        # Simple deterministic evolutionary approach with better control
        best_c2 = 0.0
        best_individual = self.create_individual(num_peaks)

        # Start with a few iterations of hill climbing
        for iteration in range(100):
            current_individual = list(best_individual)

            # Mutate randomly
            mutated_individual = list(current_individual)

            # Select a parameter to mutate
            param_index = random.randint(0, len(mutated_individual) - 1)
            # Apply mutation with appropriate scaling
            if param_index % 3 == 0:  # amplitude
                mutated_individual[param_index] *= random.uniform(0.8, 1.2)
            elif param_index % 3 == 1:  # center
                mutated_individual[param_index] += random.uniform(-0.01, 0.01)
            else:  # width
                mutated_individual[param_index] *= random.uniform(0.9, 1.1)

            # Ensure constraints
            mutated_individual[param_index] = max(0, mutated_individual[param_index])

            # Enforce peak spacing after mutation
            self.enforce_peak_spacing(mutated_individual)

            # Evaluate
            c2_value = self.evaluate_individual(mutated_individual)

            if c2_value > best_c2:
                best_c2 = c2_value
                best_individual = list(mutated_individual)

                # Early termination if improvement is significant
                if iteration > 10 and c2_value > 0.95:
                    break

        # Final refinement with focused search
        for _ in range(50):
            refined_individual = list(best_individual)
            param_index = random.randint(0, len(refined_individual) - 1)

            # Fine tuning
            if param_index % 3 == 0:  # amplitude
                refined_individual[param_index] *= random.uniform(0.95, 1.05)
            elif param_index % 3 == 1:  # center
                refined_individual[param_index] += random.uniform(-0.005, 0.005)
            else:  # width
                refined_individual[param_index] *= random.uniform(0.98, 1.02)

            refined_individual[param_index] = max(0, refined_individual[param_index])
            self.enforce_peak_spacing(refined_individual)

            c2_value = self.evaluate_individual(refined_individual)

            if c2_value > best_c2:
                best_c2 = c2_value
                best_individual = list(refined_individual)

        return best_individual

    def gaussian_step_function(self, n: int, sigma: float = 0.1) -> List[float]:
        """Generate a Gaussian-based step function with specified number of steps."""
        x = np.linspace(-0.25, 0.25, n, endpoint=False)
        y = np.exp(-0.5 * (x/sigma)**2)
        y = y / np.max(y) * 20
        return [float(val) for val in y]

    def adaptive_refinement(self, f_values: List[float], max_iterations: int = 300) -> List[float]:
        """Apply adaptive refinement to improve step function based on C2 value."""
        current_f = list(f_values)
        current_c2 = self.compute_c2(current_f)

        improvement_count = 0
        step_size = 0.1
        prev_c2 = current_c2

        for iteration in range(max_iterations):
            modified_f = list(current_f)
            idx = np.random.randint(len(modified_f))

            delta = np.random.uniform(-step_size, step_size)
            modified_f[idx] = max(0.0, modified_f[idx] + delta)

            test_c2 = self.compute_c2(modified_f)

            if test_c2 > current_c2:
                current_f = modified_f
                current_c2 = test_c2
                improvement_count = 0
            else:
                improvement_count += 1

            # Adjust step size
            if improvement_count > 5:
                step_size *= 0.9
                improvement_count = 0
            elif improvement_count == 0:
                step_size = min(0.1, step_size * 1.1)

            # Early stopping condition
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
        best_c2 = 0.0
        best_function = []

        # Strategy 1: Multi-start with evolutionary algorithm using different peak counts
        peak_counts = [3, 5, 7, 10, 15]

        for num_peaks in peak_counts:
            try:
                peak_params = self.optimize_with_evolutionary_algorithm(
                    num_peaks=num_peaks,
                    population_size=30,
                    generations=30
                )

                domain_points = np.linspace(-self.domain_width/2, self.domain_width/2, self.default_step_count)
                func_values = self.gaussian_peak_function(domain_points, peak_params)
                step_values = func_values.tolist()
                c2_val = self.compute_c2(step_values)

                if c2_val > best_c2:
                    best_c2 = c2_val
                    best_function = step_values

            except Exception as e:
                continue

        # Strategy 2: Fallback to adaptive refinement on Gaussian-based function
        if len(best_function) == 0:
            try:
                n_steps = np.random.randint(200, 1000)
                base_f = self.gaussian_step_function(n_steps)
                refined_f = self.adaptive_refinement(base_f, max_iterations=500)
                final_c2 = self.compute_c2(refined_f)

                if final_c2 > best_c2:
                    best_c2 = final_c2
                    best_function = refined_f
            except Exception as e:
                pass

        # Strategy 3: Final fallback to simple uniform distribution
        if len(best_function) == 0:
            best_function = [10.0] * 200

        return best_function

def construct_function() -> List[float]:
    """Main entry point function for constructing step function with high C2 value."""
    optimizer = StepFunctionOptimizer()
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")