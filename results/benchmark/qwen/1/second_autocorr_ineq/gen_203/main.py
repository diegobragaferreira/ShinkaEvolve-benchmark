# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import time
from scipy.optimize import differential_evolution, minimize
from typing import List, Tuple, Optional
import random
import math

# Core Calculation Module with Enhanced Performance
class AutoconvolutionCalculator:
    """Computes autoconvolutions and C2 norms efficiently using Numba JIT compilation"""
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def compute_autoconvolution_parallel(f_vals: np.ndarray) -> np.ndarray:
        """Efficiently compute autoconvolution with parallel processing"""
        n = len(f_vals)
        g = np.zeros(2 * n - 1)
        
        # Parallelized convolution loop for maximum performance
        for i in prange(n):
            for j in range(n):
                idx = i + j
                if 0 <= idx < len(g):
                    g[idx] += f_vals[i] * f_vals[j]
        
        return g
    
    @staticmethod
    @jit(nopython=True)
    def compute_norms(g_vals: np.ndarray) -> Tuple[float, float, float]:
        """Compute L1, L2^2, and L-infinity norms efficiently"""
        l1_norm = 0.0
        l2_sq_norm = 0.0
        linf_norm = 0.0
        
        for i in range(len(g_vals)):
            abs_val = abs(g_vals[i])
            l1_norm += abs_val
            l2_sq_norm += g_vals[i] * g_vals[i]
            if abs_val > linf_norm:
                linf_norm = abs_val
        
        return l1_norm, l2_sq_norm, linf_norm
    
    @classmethod
    def compute_c2(cls, f_vals: np.ndarray) -> float:
        """Compute C2 value with numerical safety checks"""
        try:
            # Compute autoconvolution
            g_vals = cls.compute_autoconvolution_parallel(f_vals)
            
            # Compute norms
            l1, l2_sq, linf = cls.compute_norms(g_vals)
            
            # Avoid division by zero - critical safety check
            if l1 <= 1e-15 or linf <= 1e-15:
                return 0.0
            
            # Return C2 value
            return l2_sq / (l1 * linf)
        except Exception:
            return 0.0

# Advanced Initialization Module
class AdvancedInitializer:
    """Creates diverse and mathematically informed initial step function configurations"""
    
    @staticmethod
    def create_mathematically_informed_pattern(n_steps: int) -> np.ndarray:
        """
        Create a pattern based on mathematical insights:
        - Peaks at center with gradual tapering
        - Balanced high/low regions
        - Smooth transitions to avoid numerical artifacts
        """
        # Create a symmetric pattern with mathematical properties
        pattern = np.zeros(n_steps)
        
        # Central peak with controlled width
        center = n_steps // 2
        peak_width = max(2, n_steps // 8)
        
        # Create smooth Gaussian-like peak in center
        for i in range(max(0, center - peak_width), min(n_steps, center + peak_width)):
            distance_from_center = abs(i - center)
            pattern[i] = np.exp(-0.5 * (distance_from_center / (peak_width/2))**2)
        
        # Add some structured variation - create alternating high/low segments
        segment_size = max(1, n_steps // 16)
        
        # Alternate between high and low regions
        for i in range(0, n_steps, segment_size):
            end_idx = min(i + segment_size, n_steps)
            if (i // segment_size) % 2 == 0:
                # High region - boost values
                pattern[i:end_idx] = pattern[i:end_idx] * 1.5 + 0.5
            else:
                # Low region - reduce values but keep positive
                pattern[i:end_idx] = pattern[i:end_idx] * 0.7 + 0.1
        
        # Add subtle modulation to break symmetry
        modulation = 1.0 + 0.1 * np.sin(np.linspace(0, 8*np.pi, n_steps))
        pattern = pattern * modulation
        
        # Ensure non-negativity and scale appropriately
        pattern = np.clip(pattern, 0, np.inf)
        if np.sum(pattern) > 0:
            pattern = pattern / np.sum(pattern) * n_steps
            
        return pattern
    
    @staticmethod
    def create_balanced_distribution_pattern(n_steps: int) -> np.ndarray:
        """
        Create pattern with balanced distribution of high and low values
        designed to create favorable convolution properties
        """
        # Create a pattern that tries to balance energy distribution
        pattern = np.ones(n_steps)
        
        # Introduce strategic high/low variations
        high_regions = n_steps // 8
        for i in range(high_regions):
            start_idx = i * (n_steps // high_regions)
            end_idx = min((i + 1) * (n_steps // high_regions), n_steps)
            
            if i % 2 == 0:
                # High regions
                pattern[start_idx:end_idx] = 2.0
            else:
                # Low regions
                pattern[start_idx:end_idx] = 0.3
        
        # Add some randomness for exploration while maintaining structure
        noise = np.random.normal(0, 0.1, n_steps)
        pattern = pattern + noise
        
        # Ensure non-negativity and scale
        pattern = np.clip(pattern, 0, np.inf)
        if np.sum(pattern) > 0:
            pattern = pattern / np.sum(pattern) * n_steps
            
        return pattern
    
    @staticmethod
    def create_energy_concentrated_pattern(n_steps: int) -> np.ndarray:
        """
        Create pattern focused on concentrating energy in central regions
        to promote strong convolution peaks
        """
        pattern = np.zeros(n_steps)
        
        # Create highly concentrated central region
        center = n_steps // 2
        width = max(2, n_steps // 10)
        
        # Create sharp central peak
        for i in range(max(0, center - width), min(n_steps, center + width)):
            distance = abs(i - center)
            if distance <= width:
                pattern[i] = 1.0 + (width - distance) / width * 0.5
        
        # Add some surrounding structure
        for i in range(max(0, center - width*2), min(n_steps, center + width*2)):
            if i < center - width or i > center + width:
                distance = min(abs(i - center + width), abs(i - center - width))
                if distance <= width:
                    pattern[i] = 0.1 + (width - distance) / width * 0.3
        
        # Add controlled noise to break perfect symmetry
        noise = 0.05 * np.random.normal(0, 1, n_steps)
        pattern = pattern + noise
        
        # Ensure non-negativity and normalize
        pattern = np.clip(pattern, 0, np.inf)
        if np.sum(pattern) > 0:
            pattern = pattern / np.sum(pattern) * n_steps
            
        return pattern

    @classmethod
    def create_multi_scale_initialization(cls, n_steps: int) -> np.ndarray:
        """Create diverse initial solution using multiple strategies"""
        strategies = [
            cls.create_mathematically_informed_pattern,
            cls.create_balanced_distribution_pattern,
            cls.create_energy_concentrated_pattern
        ]
        
        # Choose a strategy randomly with preference for mathematically informed ones
        strategy_weights = [0.5, 0.3, 0.2]  # Bias towards better performing patterns
        strategy = np.random.choice(strategies, p=strategy_weights)
        pattern = strategy(n_steps)
        
        # Add some randomization to make it truly diverse
        noise = np.random.normal(0, 0.05, n_steps)
        pattern = pattern + noise
        pattern = np.clip(pattern, 0, np.inf)
        
        # Normalize to maintain reasonable scale
        if np.sum(pattern) > 0:
            pattern = pattern / np.sum(pattern) * n_steps
            
        return pattern

# Advanced Optimization Engine
class AdvancedOptimizer:
    """Main optimization controller with enhanced adaptive strategies"""
    
    def __init__(self):
        self.best_solution = None
        self.best_c2 = -float('inf')
        self.max_time_seconds = 90.0
        self.seed = 42
    
    def evaluate_function(self, f_vals: List[float]) -> float:
        """Primary evaluation function with comprehensive error handling"""
        try:
            # Ensure non-negative values with fast list comprehension  
            f_vals = np.array([max(0.0, x) for x in f_vals])
            
            # Handle edge cases immediately
            if len(f_vals) == 0:
                return 0.0
            
            # Compute C2 value using optimized calculator
            c2 = AutoconvolutionCalculator.compute_c2(f_vals)
            
            # Ensure finite values
            if np.isnan(c2) or np.isinf(c2):
                return 0.0
                
            return c2
        except Exception as e:
            return 0.0
    
    def adaptive_evolutionary_optimization(self, initial_solution: List[float], 
                                        max_iter: int = 50) -> List[float]:
        """
        Enhanced evolutionary optimization with adaptive population sizing
        and convergence detection
        """
        # Track convergence
        best_scores = []
        patience_counter = 0
        max_patience = 5
        population_size = 15  # Starting population size
        
        # Time tracking
        start_time = time.time()
        
        # Start with initial solution
        current_solution = initial_solution.copy()
        current_c2 = self.evaluate_function(current_solution)
        best_scores.append(current_c2)
        
        # Adaptive algorithm parameters
        for generation in range(max_iter):
            # Early termination check
            if time.time() - start_time > self.max_time_seconds * 0.9:
                break
            
            # Check for convergence
            if len(best_scores) >= 3:
                recent_improvement = best_scores[-1] - best_scores[-3]
                if recent_improvement < 1e-8:
                    patience_counter += 1
                else:
                    patience_counter = 0
                
                # Increase population size if stuck
                if patience_counter >= max_patience:
                    population_size = min(population_size * 2, 30)
                    patience_counter = 0
            
            # Define bounds for differential evolution
            bounds = [(0.0, 10.0)] * len(current_solution)
            
            try:
                # Run differential evolution with adaptive parameters
                result = differential_evolution(
                    lambda x: -self.evaluate_function(x),  # Negative for maximization
                    bounds,
                    maxiter=3,  # Fewer iterations per generation for speed
                    popsize=population_size,
                    seed=self.seed + generation,
                    strategy='best1bin',
                    tol=1e-6,
                    recombination=0.7,
                    disp=False
                )
                
                if result.success:
                    new_solution = result.x.tolist()
                    new_c2 = self.evaluate_function(new_solution)
                    
                    if new_c2 > current_c2:
                        current_solution = new_solution
                        current_c2 = new_c2
                        best_scores.append(current_c2)
                        
            except Exception:
                pass  # Continue with current solution if optimization fails
        
        return current_solution
    
    def local_refinement(self, solution: List[float], max_iter: int = 20) -> List[float]:
        """
        Apply local refinement to improve solution quality
        """
        refined_solution = solution.copy()
        
        # Try different local search strategies
        for iteration in range(max_iter):
            try:
                # Use scipy.optimize.minimize for local refinement
                x0 = np.array(refined_solution[:min(len(refined_solution), 200)])
                bounds = [(0, 10.0)] * len(x0)
                
                def objective(x):
                    # Extend to full size
                    extended_x = list(x) + [1.0] * (len(solution) - len(x))
                    return -self.evaluate_function(extended_x)
                
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 10})
                
                if res.success:
                    refined_solution = np.maximum(res.x, 0)
                    # Extend back to original size
                    extended_refined = list(refined_solution) + [1.0] * (len(solution) - len(refined_solution))
                    refined_solution = extended_refined
            except Exception:
                pass
            
            # Small stochastic perturbation to escape local minima
            if iteration % 3 == 0:
                for i in range(len(refined_solution)):
                    # Add small random noise occasionally
                    if np.random.random() < 0.1:
                        noise = np.random.normal(0, 0.01)  # Smaller magnitude noise
                        refined_solution[i] = max(0, refined_solution[i] + noise)
        
        return refined_solution
    
    def multi_scale_search(self) -> List[float]:
        """Enhanced multi-scale search with adaptive exploration"""
        best_solution = None
        best_c2 = -float('inf')
        start_time = time.time()
        
        # Multiple search phases with different intensities
        search_phases = [
            {"attempts": 5, "max_steps_range": (100, 300), "iterations": 30},
            {"attempts": 8, "max_steps_range": (300, 800), "iterations": 40},
            {"attempts": 5, "max_steps_range": (800, 1000), "iterations": 50}
        ]
        
        for phase_info in search_phases:
            attempts = phase_info["attempts"]
            min_steps, max_steps = phase_info["max_steps_range"]
            iterations = phase_info["iterations"] 
            
            for attempt in range(attempts):
                # Early termination check
                if time.time() - start_time > self.max_time_seconds * 0.95:
                    break
                    
                # Create diverse initial solution
                n_steps = np.random.randint(min_steps, max_steps)
                initial_solution = AdvancedInitializer.create_multi_scale_initialization(n_steps)
                
                # Apply adaptive evolutionary optimization
                evolved_solution = self.adaptive_evolutionary_optimization(initial_solution, iterations)
                
                # Apply local refinement
                refined_solution = self.local_refinement(evolved_solution)
                
                # Evaluate result
                c2 = self.evaluate_function(refined_solution)
                
                if c2 > best_c2:
                    best_c2 = c2
                    best_solution = refined_solution
        
        return best_solution if best_solution is not None else [1.0] * 100

# Main Controller
def construct_function() -> List[float]:
    """
    Main entry point for constructing step-function with high C2 value.
    Uses advanced modular optimization approach with adaptive strategies.
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize optimizer
    optimizer = AdvancedOptimizer()
    
    try:
        # Use enhanced multi-scale search approach
        best_solution = optimizer.multi_scale_search()
        
        # Final evaluation
        final_c2 = optimizer.evaluate_function(best_solution)
        
        end_time = time.time()
        eval_time = end_time - time.time()  # This was incorrectly set to start_time previously
        
        print(f"Eval time: {eval_time:.4f}s")
        print(f"Best C2 found: {final_c2:.6f}")
        
        return best_solution
        
    except Exception as e:
        # Fallback to simple approach if optimization fails
        print(f"Optimization failed with error: {e}. Using fallback.")
        fallback_solution = [1.0] * 100
        return fallback_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")