# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
import time
from numba import jit, prange
from typing import List, Tuple, Optional, Callable

# Core JIT-compiled functions for performance
@jit(nopython=True)
def compute_autoconvolution_fast(f: np.ndarray) -> np.ndarray:
    """Fast autoconvolution computation using numba."""
    n = len(f)
    g = np.zeros(2*n - 1, dtype=np.float64)

    # Manual convolution loop for speed
    for i in range(n):
        for j in range(n):
            g[i + j] += f[i] * f[j]

    return g[n-1:]  # Return positive lags only

@jit(nopython=True)
def compute_norms_fast(g: np.ndarray) -> Tuple[float, float, float]:
    """Fast norm computations using numba."""
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
def compute_c2_fast(norm_1: float, norm_2_sq: float, norm_inf: float) -> float:
    """Fast C2 computation using numba."""
    if norm_1 < 1e-12 or norm_inf < 1e-12:
        return 0.0
    return norm_2_sq / (norm_1 * norm_inf)

class StepFunctionStrategy:
    """Encapsulates different strategies for generating step functions."""

    @staticmethod
    def uniform_distribution(n_steps: int) -> np.ndarray:
        """Generate uniform distribution function."""
        return np.ones(n_steps) * 0.5

    @staticmethod
    def improved_gaussian_peaks(n_steps: int, seed: int) -> np.ndarray:
        """Generate improved Gaussian peak structure with better spacing and adaptive characteristics."""
        np.random.seed(seed)
        x = np.linspace(-0.25, 0.25, n_steps)
        base_function = np.zeros_like(x)

        # Use logarithmic spacing for peak positions with better distribution
        num_peaks = np.random.randint(15, 25)  # Increased peak count for better resolution

        # More sophisticated logarithmic distribution that ensures better coverage
        log_min = np.log10(0.005)  # Tighter minimum spacing
        log_max = np.log10(0.24)
        log_positions = np.logspace(log_min, log_max, num_peaks)

        # Mirror the positions to both sides
        peak_positions = np.concatenate([log_positions, -log_positions[::-1]])
        peak_positions = peak_positions[(peak_positions <= 0.25) & (peak_positions >= -0.25)]

        # Ensure minimum spacing between peaks (more aggressive spacing)
        min_gap = 0.03  # Tighter minimum gap
        safe_positions = []
        for pos in sorted(peak_positions):
            if not safe_positions or abs(pos - safe_positions[-1]) >= min_gap:
                safe_positions.append(pos)

        num_peaks = len(safe_positions)

        # Construct peaks with more systematic height and width selection
        for i in range(num_peaks):
            peak_center = safe_positions[i]

            # Adaptive peak height based on position
            # Central peaks get higher amplitude to create stronger autoconvolution signals
            if abs(peak_center) < 0.05:
                peak_height = np.random.uniform(1.5, 2.5)  # Higher amplitude for center peaks
            elif abs(peak_center) < 0.15:
                peak_height = np.random.uniform(1.2, 2.0)  # Medium amplitude for mid-range
            else:
                peak_height = np.random.uniform(0.8, 1.5)  # Lower amplitude for outer peaks

            # Adaptive peak width based on position - wider for outer peaks to avoid overly sharp peaks
            if abs(peak_center) < 0.05:
                peak_width = np.random.uniform(0.012, 0.025)  # Narrower for center
            elif abs(peak_center) < 0.15:
                peak_width = np.random.uniform(0.020, 0.040)  # Medium width
            else:
                peak_width = np.random.uniform(0.035, 0.060)  # Wider for outer peaks

            # Create Gaussian peak
            gaussian_peak = peak_height * np.exp(-0.5 * ((x - peak_center) / peak_width)**2)
            base_function += gaussian_peak

        # Add additional structure with more controlled randomness
        # This creates smoother transitions between peaks
        for i in range(0, len(x), max(1, len(x)//30)):  # Less frequent bumps but more controlled
            if np.random.random() > 0.85:  # Slightly reduced bump rate
                bump_center = x[i]
                bump_height = np.random.uniform(0.03, 0.15)  # Lower amplitude bumps
                bump_width = np.random.uniform(0.008, 0.020)  # Narrow bumps
                bump = bump_height * np.exp(-0.5 * ((x - bump_center) / bump_width)**2)
                base_function += bump

        # Add a more structured component - a few larger, broader peaks for better autoconvolution
        if num_peaks > 10:
            # Add 2-3 larger structure peaks
            structure_positions = [-0.12, 0.0, 0.12]
            structure_heights = [1.0, 1.8, 1.0]  # Central peak is highest
            structure_widths = [0.04, 0.08, 0.04]  # Wider central peak

            for center, height, width in zip(structure_positions, structure_heights, structure_widths):
                if -0.25 <= center <= 0.25:
                    structure_peak = height * np.exp(-0.5 * ((x - center) / width)**2)
                    base_function += structure_peak

        return base_function

class PerformanceMonitor:
    """Monitors performance and manages time constraints."""

    def __init__(self, max_time: float = 90.0):
        self.max_time = max_time
        self.start_time = time.time()

    def time_remaining(self) -> float:
        """Returns remaining time in seconds."""
        return self.max_time - (time.time() - self.start_time)

    def is_expired(self) -> bool:
        """Check if time budget is exhausted."""
        return self.time_remaining() <= 0.1

class AutoCorrelationEvaluator:
    """Handles evaluation of functions and computation of C2."""

    @staticmethod
    def evaluate_function(f: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate function and compute C2."""
        try:
            # Fast autoconvolution
            g = compute_autoconvolution_fast(f)

            # Fast norm computations
            norm_1, norm_2_sq, norm_inf = compute_norms_fast(g)

            # C2 computation
            c2 = compute_c2_fast(norm_1, norm_2_sq, norm_inf)

            return c2, g
        except Exception:
            return 0.0, np.array([0.0])

class OptimizationPipeline:
    """Main optimization pipeline orchestrating the search process."""

    def __init__(self, seed: int = 42, max_time: float = 90.0):
        self.seed = seed
        self.performance_monitor = PerformanceMonitor(max_time)
        self.evaluator = AutoCorrelationEvaluator()

        # Initialize deterministic behavior
        np.random.seed(seed)
        random.seed(seed)

    def _construct_initial_functions(self, n_steps: int) -> List[Tuple[np.ndarray, str]]:
        """Construct various initial candidate functions with better prioritization."""
        functions = []

        # Strategy 1: Improved Gaussian peaks (most promising)
        if self.performance_monitor.time_remaining() > 15.0:
            improved_gaussian_func = StepFunctionStrategy.improved_gaussian_peaks(n_steps, self.seed)
            functions.append((improved_gaussian_func, "improved_gaussian"))

        # Strategy 2: Uniform distribution (fallback)
        if self.performance_monitor.time_remaining() > 5.0:
            uniform_func = StepFunctionStrategy.uniform_distribution(n_steps)
            functions.append((uniform_func, "uniform"))

        # Strategy 3: Original Gaussian peaks (backup)
        if self.performance_monitor.time_remaining() > 10.0:
            gaussian_func = StepFunctionStrategy.gaussian_peaks(n_steps, self.seed)
            functions.append((gaussian_func, "gaussian"))

        return functions

    def _enhance_function(self, base_function: np.ndarray, n_steps: int) -> np.ndarray:
        """Apply enhancements to improve function quality."""
        # Ensure non-negative values
        enhanced_function = np.maximum(base_function, 0)

        # Normalize
        if np.max(enhanced_function) > 0:
            enhanced_function = enhanced_function / np.max(enhanced_function) * 1.5

        # Apply light noise
        noise_level = 0.02
        noisy_function = enhanced_function + np.random.normal(0, noise_level, len(enhanced_function))
        noisy_function = np.maximum(noisy_function, 0)

        # Apply smoothing (fallback to direct approach if necessary)
        window_size = max(1, n_steps // 200)
        if window_size % 2 == 0:
            window_size += 1
        if window_size > 1:
            try:
                smoothed_function = signal.savgol_filter(noisy_function, window_size, 1)
                smoothed_function = np.maximum(smoothed_function, 0)
                noisy_function = smoothed_function
            except:
                pass  # Continue with original if smoothing fails

        return noisy_function

    def _hill_climb_refinement(self, f: np.ndarray) -> np.ndarray:
        """Enhanced hill climbing refinement with adaptive perturbation."""
        if self.performance_monitor.is_expired():
            return f

        current_f = f.copy()
        best_f = current_f.copy()
        best_c2, _ = self.evaluator.evaluate_function(best_f)

        # Adaptive iteration count based on remaining time
        max_iterations = min(100, int(self.performance_monitor.time_remaining() * 3))

        for iteration in range(max_iterations):
            if self.performance_monitor.is_expired():
                break

            # Adaptive perturbation magnitude based on iteration progress
            perturbation_magnitude = max(0.001, 0.02 * (1.0 - iteration / max_iterations))

            # Random perturbation with more strategic selection
            perturbed_f = current_f.copy()

            # Modify more elements early, fewer later for fine-tuning
            num_modifications = max(1, int(len(current_f) * (0.05 + 0.05 * (1.0 - iteration / max_iterations))))
            indices_to_modify = np.random.choice(
                len(current_f),
                size=num_modifications,
                replace=False
            )

            for idx in indices_to_modify:
                # Add noise with adaptive magnitude
                perturbed_f[idx] += np.random.normal(0, perturbation_magnitude)
                perturbed_f[idx] = max(0, perturbed_f[idx])  # Keep non-negative

            # Evaluate perturbed function
            c2_new, _ = self.evaluator.evaluate_function(perturbed_f)

            if c2_new > best_c2:
                best_c2 = c2_new
                best_f = perturbed_f.copy()
                current_f = perturbed_f.copy()
            else:
                # Simulated annealing with temperature decay
                temperature = 0.1 * (1.0 - iteration / max_iterations)
                if np.random.random() < np.exp((c2_new - best_c2) / (temperature + 1e-10)):
                    current_f = perturbed_f.copy()

        return best_f

    def _optimize_with_differential_evolution(self, initial_func: np.ndarray, n_steps: int) -> np.ndarray:
        """Enhanced differential evolution optimization with better time management."""
        if self.performance_monitor.is_expired() or self.performance_monitor.time_remaining() < 8.0:
            return initial_func

        try:
            x = np.linspace(-0.25, 0.25, n_steps)

            # Identify approximate peak locations with better detection
            peaks = []
            for i in range(1, len(initial_func)-1):
                if initial_func[i] > initial_func[i-1] and initial_func[i] > initial_func[i+1]:
                    peaks.append((i, initial_func[i]))

            # Early exit if few peaks
            if len(peaks) == 0:
                return initial_func

            # Sort by height and take top peaks with better selection logic
            peaks.sort(key=lambda x: x[1], reverse=True)
            selected_peaks = peaks[:min(12, len(peaks))]  # More peaks for better optimization

            if len(selected_peaks) == 0:
                return initial_func

            # Objective function for optimization with better bounds
            def objective(params):
                if self.performance_monitor.is_expired():
                    return 1e10

                temp_func = np.zeros_like(x)
                param_idx = 0
                for i, (pos_idx, height) in enumerate(selected_peaks):
                    if param_idx + 1 >= len(params):
                        break
                    # Broader bounds for better exploration
                    center_shift = (params[param_idx] - 0.5) * 0.1  # Larger shift range
                    center_pos = x[pos_idx] + center_shift
                    peak_height = height * (1.0 + params[param_idx + 1] * 0.8)  # Larger height variation
                    # Width varies more to explore different peak shapes
                    width = 0.02 + params[param_idx + 2] * 0.05  # Variable width
                    temp_func += peak_height * np.exp(-0.5 * ((x - center_pos) / width)**2)
                    param_idx += 3  # Now use 3 parameters per peak

                # Return negative C2 (minimization)
                c2_value, _ = self.evaluator.evaluate_function(temp_func)
                return -c2_value if c2_value > 0 else 1e10

            # Initial parameter guess (now 3 parameters per peak)
            params0 = [0.0] * (len(selected_peaks) * 3)

            # Dynamic iteration count based on time and problem size
            time_available = self.performance_monitor.time_remaining()
            maxiter = min(50, int(time_available * 2))  # More iterations for more time
            if maxiter < 10:
                return initial_func

            # Population size should be larger for better exploration when time permits
            popsize = min(15, max(5, int(maxiter * 0.6)))

            # Optimization with more robust settings
            try:
                result = differential_evolution(
                    objective,
                    bounds=[(-0.8, 0.8)] * (len(selected_peaks) * 3),  # Broader bounds
                    maxiter=maxiter,
                    popsize=popsize,
                    seed=self.seed,
                    polish=True,
                    disp=False,
                    mutation=(0.5, 1.0),  # Better mutation strategy
                    recombination=0.7  # Better crossover rate
                )

                if result.success:
                    optimized_params = result.x

                    # Apply optimization results
                    final_func = np.zeros_like(x)
                    param_idx = 0
                    for i, (pos_idx, height) in enumerate(selected_peaks):
                        if param_idx + 2 >= len(optimized_params):
                            break
                        center_shift = (optimized_params[param_idx] - 0.5) * 0.1
                        center_pos = x[pos_idx] + center_shift
                        peak_height = height * (1.0 + optimized_params[param_idx + 1] * 0.8)
                        width = 0.02 + optimized_params[param_idx + 2] * 0.05
                        final_func += peak_height * np.exp(-0.5 * ((x - center_pos) / width)**2)
                        param_idx += 3

                    # Incorporate remaining components with better blending
                    for i in range(len(initial_func)):
                        if not any(abs(x[i] - x[pos_idx]) < 0.01 for _, pos_idx in selected_peaks):
                            final_func[i] += initial_func[i] * 0.3  # Slightly more weight to original

                    return final_func
            except Exception as e:
                # If DE fails, fall back to a simpler approach
                pass

        except Exception as e:
            # General exception handling
            pass

        return initial_func

    def _execute_strategy(self, strategy_name: str, f: np.ndarray) -> Tuple[float, np.ndarray]:
        """Execute a specific optimization strategy."""
        if self.performance_monitor.is_expired():
            return 0.0, f

        # Apply enhancement
        enhanced_f = self._enhance_function(f, len(f))

        # Refinement phase
        refined_f = self._hill_climb_refinement(enhanced_f)

        # Differential evolution optimization (when time allows)
        if strategy_name == "gaussian":
            optimized_f = self._optimize_with_differential_evolution(refined_f, len(refined_f))
        else:
            optimized_f = refined_f

        # Final evaluation
        c2, g = self.evaluator.evaluate_function(optimized_f)

        return c2, optimized_f

    def construct_function(self) -> List[float]:
        """Main function to construct step-function with high C2 value."""
        if self.performance_monitor.is_expired():
            return [0.5] * 1000

        # Determine number of steps
        n_steps = np.random.randint(1000, 10000)

        # Early exit if insufficient time
        if self.performance_monitor.time_remaining() < 10.0:
            return [0.5] * 1000

        # Generate initial functions using different strategies
        initial_functions = self._construct_initial_functions(n_steps)

        # Evaluate all strategies and select the best
        best_c2 = 0.0
        best_f = None

        for func, strategy_name in initial_functions:
            if self.performance_monitor.is_expired():
                break

            try:
                c2, evaluated_f = self._execute_strategy(strategy_name, func)

                if c2 > best_c2:
                    best_c2 = c2
                    best_f = evaluated_f.copy()

            except Exception:
                continue

        # Fallback to default if no good solution found
        if best_f is None or best_c2 <= 0:
            default_func = StepFunctionStrategy.uniform_distribution(n_steps)
            best_c2, best_f = self.evaluator.evaluate_function(default_func)

        # Final post-processing and noise addition
        final_result = np.maximum(best_f, 0).tolist()

        # Add slight noise for robustness
        noise_level = 0.01
        noisy_func = np.array(final_result) + np.random.normal(0, noise_level, len(final_result))
        noisy_func = np.maximum(noisy_func, 0)

        return noisy_func.tolist()

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    optimizer = OptimizationPipeline()
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")