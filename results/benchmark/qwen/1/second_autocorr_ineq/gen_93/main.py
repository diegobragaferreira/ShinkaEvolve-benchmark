# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import convolve
import time
from typing import List, Tuple, Optional
import warnings

class NormCalculator:
    """Computes autoconvolution norms with numerical stability optimizations."""
    
    @staticmethod
    def compute_autoconvolution_norms(f_values: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute the three norms needed for C2 calculation:
        ||g||₂² (L2 norm squared), ||g||₁ (L1 norm), ||g||∞ (L-infinity norm)
        
        Args:
            f_values: array of step function heights
            
        Returns:
            tuple: (norm_g2_sq, norm_g1, norm_ginf)
        """
        # Ensure non-negative values
        f = np.maximum(f_values, 0.0)
        
        # Compute autoconvolution g = f * f (discrete convolution)
        g = convolve(f, f, mode='full')
        
        # Extract the central portion that represents the main interval
        n = len(f)
        middle_idx = n - 1
        half_width = n
        
        # Take the central part of the convolution
        g_centered = g[middle_idx - half_width + 1 : middle_idx + half_width]
        
        # Compute the norms
        g_squared = g_centered ** 2
        g_abs = np.abs(g_centered)
        
        # ||g||₂² - sum of squares
        norm_g2_sq = np.sum(g_squared)
        
        # ||g||₁ - sum of absolute values
        norm_g1 = np.sum(g_abs)
        
        # ||g||∞ - maximum absolute value
        norm_ginf = np.max(g_abs)
        
        return norm_g2_sq, norm_g1, norm_ginf

class StepFunction:
    """Represents a step function with utility methods for optimization."""
    
    def __init__(self, values: List[float]):
        self.values = np.array(values, dtype=np.float64)
        self._validate()
        
    def _validate(self):
        """Validate that step function values are finite and non-negative."""
        if not np.all(np.isfinite(self.values)):
            raise ValueError("Step function contains infinite or NaN values")
        if np.any(self.values < 0):
            warnings.warn("Negative values found, clipping to zero")
            self.values = np.maximum(self.values, 0.0)
    
    def get_values(self) -> np.ndarray:
        """Get the current step function values."""
        return self.values.copy()
    
    def set_values(self, new_values: List[float]):
        """Set new step function values."""
        self.values = np.array(new_values, dtype=np.float64)
        self._validate()
    
    def normalize(self):
        """Normalize the step function to prevent extreme values."""
        max_val = np.max(self.values)
        if max_val > 0:
            self.values = self.values / max_val
    
    def smooth(self, kernel_size: int = None):
        """Apply Gaussian smoothing to the step function."""
        if len(self.values) < 3:
            return
            
        if kernel_size is None:
            kernel_size = min(21, len(self.values) // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        sigma = kernel_size / 6.0
        x = np.arange(kernel_size) - kernel_size // 2
        gaussian_kernel = np.exp(-x**2 / (2 * sigma**2))
        gaussian_kernel /= np.sum(gaussian_kernel)
        
        # Apply convolution to smooth the function
        self.values = np.convolve(self.values, gaussian_kernel, mode='same')
        self.values = np.maximum(self.values, 0.0)

class EvolutionaryOptimizer:
    """Main optimization engine with adaptive strategies."""
    
    def __init__(self, max_time_seconds: float = 90.0, seed: int = 42):
        self.max_time_seconds = max_time_seconds
        self.seed = seed
        self.norm_calculator = NormCalculator()
        
    def _create_initial_population(self, n_steps: int, population_size: int = 10) -> List[np.ndarray]:
        """Generate diverse initial population based on mathematical insights."""
        population = []
        
        for _ in range(population_size):
            # Strategy selection
            strategy = np.random.choice([
                'uniform', 'alternating', 'gaussian', 'mixed'
            ])
            
            if strategy == 'uniform':
                values = [0.5] * n_steps
            elif strategy == 'alternating':
                values = [np.random.uniform(0.7, 1.0) if i % 2 == 0 
                         else np.random.uniform(0.0, 0.3) for i in range(n_steps)]
            elif strategy == 'gaussian':
                x = np.linspace(-1, 1, n_steps)
                mu, sigma = 0, 0.3
                gauss = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                scale_factor = 1.0 / np.max(gauss)
                values = (gauss * scale_factor).tolist()
            else:  # mixed
                values = [np.random.uniform(0.7, 1.0) if i % 2 == 0 
                         else np.random.uniform(0.0, 0.3) for i in range(n_steps)]
                # Apply Gaussian smoothing
                if n_steps >= 5:
                    kernel_size = min(11, n_steps // 10)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    sigma = kernel_size / 6.0
                    x = np.arange(kernel_size) - kernel_size // 2
                    gaussian_kernel = np.exp(-x**2 / (2 * sigma**2))
                    gaussian_kernel /= np.sum(gaussian_kernel)
                    values = np.convolve(values, gaussian_kernel, mode='same').tolist()
            
            # Add noise for exploration
            noise_level = 0.05
            values = [max(0, val + np.random.normal(0, noise_level)) for val in values]
            
            population.append(np.array(values))
            
        return population
    
    def _evaluate_fitness(self, individual: np.ndarray) -> float:
        """Evaluate fitness (negative C2) of an individual."""
        try:
            # Ensure non-negative values
            individual = np.maximum(individual, 0.0)
            
            # Compute norms
            norm_g2_sq, norm_g1, norm_ginf = self.norm_calculator.compute_autoconvolution_norms(individual)
            
            # Avoid division by zero
            if norm_g1 < 1e-15 or norm_ginf < 1e-15:
                return -0.0  # Return worst possible fitness
            
            # C2 = ||g||₂² / (||g||₁ · ||g||∞)
            c2 = norm_g2_sq / (norm_g1 * norm_ginf)
            
            # Return negative because we want to maximize C2
            return -c2
            
        except Exception:
            return -1e10  # Very poor fitness for invalid cases
    
    def _adaptive_evolutionary_search(self, initial_individual: np.ndarray) -> np.ndarray:
        """Perform adaptive evolutionary search with multiple phases."""
        start_time = time.time()
        n_dimensions = len(initial_individual)
        
        # Phase 1: Differential Evolution (Global Search)
        bounds = [(0, 3) for _ in range(n_dimensions)]
        max_iter_phase1 = min(50, int(self.max_time_seconds * 0.7))
        popsize_phase1 = min(15, max(5, n_dimensions // 10))
        
        try:
            def objective(x):
                return self._evaluate_fitness(x)
                
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=max_iter_phase1,
                popsize=popsize_phase1,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=self.seed,
                disp=False
            )
            
            best_solution = de_result.x if de_result.success else initial_individual
            
        except Exception:
            best_solution = initial_individual
            
        # Phase 2: Local Refinement (Fine-tuning)
        try:
            def objective(x):
                return self._evaluate_fitness(x)
                
            refined_result = minimize(
                objective,
                best_solution,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': max(10, int(self.max_time_seconds * 0.2)),
                    'ftol': 1e-8,
                    'gtol': 1e-8
                }
            )
            
            if refined_result.success:
                final_solution = refined_result.x
            else:
                final_solution = best_solution
                
        except Exception:
            final_solution = best_solution
            
        # Post-processing
        final_solution = np.maximum(final_solution, 0.0)
        
        # Normalize to maintain reasonable scale
        total = np.sum(final_solution)
        if total > 0:
            final_solution = final_solution / total * n_dimensions * 0.5
            
        return final_solution
    
    def optimize(self, n_steps: int = None) -> np.ndarray:
        """
        Main optimization routine.
        
        Args:
            n_steps: Number of steps in the function (random if None)
            
        Returns:
            Optimized step function values
        """
        if n_steps is None:
            n_steps = np.random.randint(500, 5000)
            
        # Set seeds for reproducibility
        np.random.seed(self.seed)
        
        # Generate initial population
        initial_population = self._create_initial_population(n_steps, 5)
        
        # Select the best initial function
        best_initial = min(initial_population, key=self._evaluate_fitness)
        
        # Perform adaptive evolutionary search
        optimized_values = self._adaptive_evolutionary_search(best_initial)
        
        return optimized_values

def construct_function() -> List[float]:
    """
    Main entry point for constructing step-function with high C2 value.
    
    Returns:
        List of step function heights that maximize C2
    """
    # Create optimizer with time constraints
    optimizer = EvolutionaryOptimizer(max_time_seconds=90.0, seed=42)
    
    try:
        # Perform optimization
        optimized_function = optimizer.optimize()
    except Exception as e:
        # Fallback to simple initialization if optimization fails
        print(f"Optimization failed with error: {e}. Using fallback.")
        # Generate simple pattern
        n_steps = np.random.randint(500, 5000)
        f_values = [0.5] * n_steps
        optimized_function = np.array(f_values)
    
    # Ensure valid output format
    return optimized_function.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
