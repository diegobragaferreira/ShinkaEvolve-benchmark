# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import convolve
import time
from typing import List, Tuple, Optional
import numba
from numba import jit
from dataclasses import dataclass
from enum import Enum

@dataclass
class OptimizationResult:
    """Data class to store optimization results"""
    solution: List[float]
    score: float
    time_used: float

class InitializationStrategy(Enum):
    """Enumeration of initialization strategies"""
    BINARY = "binary"
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    MULTIPEAK = "multipeak"
    SPARSE = "sparse"
    RAMP = "ramp"

class EvaluationModule:
    """Handles all computation of autoconvolution norms and C2 evaluation"""
    
    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution_numba(f_values: np.ndarray) -> np.ndarray:
        """Fast computation of autoconvolution using numba-compiled loop"""
        n = len(f_values)
        g = np.zeros(2*n - 1)

        for i in range(n):
            for j in range(n):
                g[i + j] += f_values[i] * f_values[j]

        return g

    @classmethod
    def compute_autoconvolution_norms(cls, f_values: List[float]) -> Tuple[float, float, float]:
        """
        Compute the autoconvolution g = f*f and associated norms efficiently
        """
        try:
            # Convert to numpy array and ensure non-negative values
            f = np.array(f_values, dtype=np.float64)
            f = np.maximum(f, 0)  # Clip negative values to 0

            if len(f) == 0:
                return 0.0, 0.0, 0.0

            # Create step function on [-1/4, 1/4]
            step_width = 0.5 / len(f)

            # Autoconvolution using fast numba-compiled function
            g = cls.compute_autoconvolution_numba(f)

            # Adjust indices for proper interval mapping
            # g corresponds to [-1/2, 1/2] interval, so we map to [-1/4, 1/4]
            g_center = len(g) // 2
            half_len = len(f)
            g_trimmed = g[g_center - half_len : g_center + half_len]

            # Compute norms
            # ||g||_2^2 using trapezoidal rule for piecewise linear integration
            g_abs = np.abs(g_trimmed)
            if len(g_abs) < 2:
                norm_2_squared = 0.0
            else:
                # Trapezoidal integration formula for piecewise linear segments
                # Each segment contributes (width/3)*(y1^2 + y1*y2 + y2^2)
                widths = np.full(len(g_abs)-1, step_width)
                y1 = g_abs[:-1]
                y2 = g_abs[1:]
                norm_2_squared = np.sum(widths * (y1**2 + y1*y2 + y2**2) / 3.0)

            # ||g||_1 = sum of absolute values divided by number of elements for normalization
            norm_1 = np.sum(np.abs(g_trimmed)) / (len(g_trimmed) + 1) if len(g_trimmed) > 0 else 1e-12

            # ||g||_∞ = max absolute value
            norm_inf = np.max(np.abs(g_trimmed)) if len(g_trimmed) > 0 else 1e-12

            return norm_2_squared, norm_1, norm_inf

        except Exception as e:
            # Fallback to minimal values in case of computation errors
            return 0.0, 1e-12, 1e-12

    @classmethod
    def evaluate_c2(cls, f_values: List[float]) -> float:
        """
        Evaluate C2 = ||g||₂² / (||g||₁ · ||g||∞)
        """
        norm_2_squared, norm_1, norm_inf = cls.compute_autoconvolution_norms(f_values)

        # Prevent division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0

        c2 = norm_2_squared / (norm_1 * norm_inf)
        return c2

class InitializationModule:
    """Handles all initialization strategies for step function generation"""
    
    @staticmethod
    def create_binary_pattern(n_steps: int) -> List[float]:
        """Create binary alternating high/low pattern"""
        return [1.0 if i % 2 == 0 else 0.1 for i in range(n_steps)]

    @staticmethod
    def create_gaussian_pattern(n_steps: int) -> List[float]:
        """Create multi-scale Gaussian peaks pattern"""
        x = np.linspace(-1, 1, n_steps)
        pattern = np.zeros(n_steps)
        scales = [0.05, 0.1, 0.2]
        positions = [-0.6, -0.2, 0.2, 0.6]
        for scale in scales:
            for pos in positions:
                pattern += np.exp(-((x - pos)**2) / scale) * 0.5
        return pattern.tolist()

    @staticmethod
    def create_uniform_pattern(n_steps: int) -> List[float]:
        """Create uniform distribution pattern"""
        return [1.0] * n_steps

    @staticmethod
    def create_multipeak_pattern(n_steps: int) -> List[float]:
        """Create multi-peak pattern with varying amplitudes"""
        x = np.linspace(-1, 1, n_steps)
        pattern = np.zeros(n_steps)
        peak_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
        amplitudes = [0.8, 1.2, 0.6, 1.0, 0.9]
        for pos, amp in zip(peak_positions, amplitudes):
            pattern += np.exp(-((x - pos)**2) / 0.05) * amp
        return pattern.tolist()

    @staticmethod
    def create_sparse_pattern(n_steps: int) -> List[float]:
        """Create sparse peak pattern"""
        x = np.linspace(-1, 1, n_steps)
        pattern = np.zeros(n_steps)
        sparse_positions = [0.1, 0.5, 0.9]
        for pos in sparse_positions:
            pattern += np.exp(-((x - pos)**2) / 0.03) * 1.5
        return pattern.tolist()

    @staticmethod
    def create_ramp_pattern(n_steps: int) -> List[float]:
        """Create smooth ramp pattern"""
        x = np.linspace(-1, 1, n_steps)
        pattern = np.exp(-((x - 0.1)**2) / 0.1) + np.exp(-((x + 0.1)**2) / 0.1)
        return pattern.tolist()

    @classmethod
    def create_multiple_initializations(cls, n_steps: int) -> List[List[float]]:
        """
        Create multiple diverse initializations to enhance exploration
        """
        strategies = [
            cls.create_binary_pattern,
            cls.create_gaussian_pattern,
            cls.create_uniform_pattern,
            cls.create_multipeak_pattern,
            cls.create_sparse_pattern,
            cls.create_ramp_pattern
        ]
        
        initializations = []
        for strategy in strategies:
            try:
                initializations.append(strategy(n_steps))
            except Exception:
                # Fallback to uniform if strategy fails
                initializations.append(cls.create_uniform_pattern(n_steps))
                
        return initializations

    @classmethod
    def select_best_initialization(cls, n_steps: int, evaluation_module: EvaluationModule) -> List[float]:
        """
        Evaluate all initializations and select the best one
        """
        initializations = cls.create_multiple_initializations(n_steps)
        
        best_score = -np.inf
        best_individual = None

        for init in initializations:
            try:
                score = evaluation_module.evaluate_c2(init)
                if score > best_score:
                    best_score = score
                    best_individual = init.copy()
            except Exception:
                continue

        # Fallback to uniform distribution if none worked
        if best_individual is None:
            best_individual = cls.create_uniform_pattern(n_steps)
            
        return best_individual

class OptimizationModule:
    """Manages all evolutionary optimization operations"""
    
    def __init__(self, evaluation_module: EvaluationModule):
        self.evaluation_module = evaluation_module

    def adaptive_population_sizing(self, n_steps: int, iteration: int) -> int:
        """
        Dynamically adjust population size based on problem characteristics and iteration
        """
        base_pop = min(50, max(10, n_steps // 20))
        # Increase population size slightly in early iterations for better exploration
        if iteration < 2:
            return min(100, base_pop * 2)
        else:
            return base_pop

    def advanced_refinement_strategy(self, best_solution: List[float], n_steps: int, bounds: List[Tuple[float, float]]) -> List[float]:
        """
        Enhanced refinement that combines multiple optimization techniques
        """
        try:
            # First, try L-BFGS-B with the current solution as starting point
            def objective(x):
                return -self.evaluation_module.evaluate_c2(x)

            # Local refinement with L-BFGS-B
            ref_result = minimize(
                objective,
                best_solution,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30}
            )

            refined_solution = ref_result.x.tolist()

            # Additional refinement with Nelder-Mead if L-BFGS didn't converge well
            if self.evaluation_module.evaluate_c2(refined_solution) < 0.8 * self.evaluation_module.evaluate_c2(best_solution):
                try:
                    nm_result = minimize(
                        objective,
                        best_solution,
                        method='Nelder-Mead',
                        options={'maxiter': 20}
                    )
                    if self.evaluation_module.evaluate_c2(nm_result.x.tolist()) > self.evaluation_module.evaluate_c2(refined_solution):
                        refined_solution = nm_result.x.tolist()
                except Exception:
                    pass  # Keep previous result if Nelder-Mead fails

            return refined_solution

        except Exception:
            return best_solution

    def multi_start_differential_evolution(self, n_steps: int, bounds: List[Tuple[float, float]], 
                                         max_time_seconds: float, start_time: float) -> Optional[List[float]]:
        """
        Run multi-start differential evolution with adaptive parameters
        """
        # Set up problem with objective function
        def objective(x):
            return -self.evaluation_module.evaluate_c2(x)  # Minimize negative to maximize C2

        remaining_time = max_time_seconds - (time.time() - start_time)
        maxiter = max(100, int(remaining_time / 2))

        # Multi-start differential evolution with adaptive parameters
        best_solution = None
        best_score = -1.0

        # Try multiple random starts to improve chances of finding global optimum
        max_starts = min(8, max(2, int(remaining_time / 8)))
        for start_iter in range(max_starts):
            if time.time() - start_time > max_time_seconds - 2.0:
                break

            # Adaptive population sizing based on iteration
            popsize = self.adaptive_population_sizing(n_steps, start_iter)

            # Vary mutation rate based on iteration
            mutation_rate = max(0.3, min(1.0, 0.6 + start_iter * 0.1))

            # Randomize the seed for different starts
            np.random.seed(int(time.time()) + start_iter * 1000)

            # Run differential evolution
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=maxiter,
                    popsize=popsize,
                    mutation=(mutation_rate, 1.0),
                    recombination=0.7,
                    seed=None,  # Let it use random seed
                    disp=False
                )

                # Check if this is better
                current_score = self.evaluation_module.evaluate_c2(result.x)
                if current_score > best_score:
                    best_score = current_score
                    best_solution = result.x.tolist()
            except Exception:
                continue

        return best_solution

    def optimize(self, n_steps: int, max_time_seconds: float = 85.0) -> OptimizationResult:
        """
        Main optimization routine with enhanced features
        """
        start_time = time.time()
        
        # Define bounds for each variable (non-negative)
        bounds = [(0.0, 5.0)] * n_steps

        # Initialize with multi-scale approach
        initial_population = []
        population_size = min(50, max(10, n_steps // 20))  # Base population size

        # Create diverse starting points
        try:
            init_module = InitializationModule()
            for _ in range(population_size):
                individual = init_module.select_best_initialization(n_steps, self.evaluation_module)
                # Add controlled noise for diversity
                noise_factor = np.random.uniform(0.8, 1.2)
                individual = [max(0.0, val * noise_factor) for val in individual]
                initial_population.append(individual)
        except Exception:
            # Fallback to simple uniform initialization
            initial_population = [[1.0] * n_steps for _ in range(population_size)]

        # Run evolutionary optimization
        try:
            best_solution = self.multi_start_differential_evolution(n_steps, bounds, max_time_seconds, start_time)
            
            # If we run out of time, use the best individual from initial population
            if time.time() - start_time > max_time_seconds - 1.0:
                # Get best from initial population
                best_initial = max(initial_population, key=self.evaluation_module.evaluate_c2)
                return OptimizationResult(best_initial, self.evaluation_module.evaluate_c2(best_initial), 
                                       time.time() - start_time)

            # Advanced refinement for the best solution
            if best_solution is not None and time.time() - start_time < max_time_seconds - 3.0:
                refined_solution = self.advanced_refinement_strategy(best_solution, n_steps, bounds)
                # Use refined solution if it performs better
                if self.evaluation_module.evaluate_c2(refined_solution) > self.evaluation_module.evaluate_c2(best_solution):
                    best_solution = refined_solution

            # Return the best solution found
            if best_solution is not None:
                return OptimizationResult(best_solution, self.evaluation_module.evaluate_c2(best_solution), 
                                       time.time() - start_time)

        except Exception as e:
            # Fallback to initial population with simple random search
            try:
                best_f = None
                best_c2 = -1.0

                # Try several random solutions
                attempts = min(200, n_steps * 3)
                for _ in range(attempts):
                    if time.time() - start_time > max_time_seconds - 1.0:
                        break
                    f_values = init_module.select_best_initialization(n_steps, self.evaluation_module)
                    c2 = self.evaluation_module.evaluate_c2(f_values)
                    if c2 > best_c2:
                        best_c2 = c2
                        best_f = f_values

                if best_f is not None:
                    return OptimizationResult(best_f, best_c2, time.time() - start_time)
            except Exception:
                pass

        # Final fallback to uniform distribution
        fallback_solution = [1.0] * n_steps
        fallback_score = self.evaluation_module.evaluate_c2(fallback_solution)
        return OptimizationResult(fallback_solution, fallback_score, time.time() - start_time)

class AdaptiveModule:
    """Manages adaptive parameters and time management"""
    
    @staticmethod
    def select_problem_size() -> int:
        """Select appropriate problem size within bounds"""
        n_steps_range = [200, 3000]
        return np.random.randint(n_steps_range[0], n_steps_range[1])

def construct_function() -> List[float]:
    """
    Main function to construct optimized step-function for high C2 value
    """
    # Allow some time budget for computation (leave 5 seconds for cleanup)
    start_time = time.time()
    
    try:
        # Initialize modules
        eval_module = EvaluationModule()
        opt_module = OptimizationModule(eval_module)
        
        # Select problem size
        n_steps = AdaptiveModule.select_problem_size()
        
        # Run optimization with time constraint
        result = opt_module.optimize(n_steps, max_time_seconds=85.0)
        
        # Final validation and cleanup
        f_values = np.array(result.solution, dtype=np.float64)
        f_values = np.maximum(f_values, 0)  # Ensure non-negative
        f_values = f_values.tolist()

        # If too long, truncate to reasonable size but maintain minimum length
        if len(f_values) > 5000:
            f_values = f_values[:5000]
        elif len(f_values) < 100:
            f_values = f_values + [0.0] * (100 - len(f_values))

        return f_values

    except Exception as e:
        # Return a fallback solution in case of any failure
        n_steps = 500
        return [1.0] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")