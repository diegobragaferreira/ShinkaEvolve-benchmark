# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import List, Tuple, Callable, Optional
import numba
from numba import jit
import warnings
import time
import random
warnings.filterwarnings('ignore')

class AutoconvolutionEvaluator:
    """Handles all autoconvolution and norm computations"""
    
    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution_fast(f_values: np.ndarray) -> np.ndarray:
        """Fast JIT-compiled autoconvolution computation"""
        n = len(f_values)
        g_length = 2 * n - 1
        g = np.zeros(g_length)
        
        for i in range(n):
            for j in range(n):
                g[i + j] += f_values[i] * f_values[j]
        
        return g
    
    @staticmethod
    def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
        """
        Compute the norms ||g||₂², ||g||₁, and ||g||∞ for the autoconvolution g = f*f
        """
        f = np.array(f_values, dtype=np.float64)
        g = AutoconvolutionEvaluator.compute_autoconvolution_fast(f)
        
        # ||g||₂² using piecewise linear integration
        norm_g_2_squared = 0.0
        for i in range(len(g) - 1):
            y1 = g[i]
            y2 = g[i + 1]
            norm_g_2_squared += (y1 * y1 + y1 * y2 + y2 * y2) / 3.0
        
        # ||g||₁ = sum(|g[i]|)
        norm_g_1 = np.sum(np.abs(g))
        
        # ||g||∞ = max(|g[i]|)
        norm_g_inf = np.max(np.abs(g))
        
        return norm_g_2_squared, norm_g_1, norm_g_inf

class C2ScoreCalculator:
    """Computes C2 score from autoconvolution norms"""
    
    @staticmethod
    def calculate_c2(f_values: List[float]) -> float:
        """Calculate C₂ = ||g||₂² / (||g||₁ · ||g||∞)"""
        try:
            norm_g_2_squared, norm_g_1, norm_g_inf = \
                AutoconvolutionEvaluator.compute_autoconvolution_norms(f_values)

            if norm_g_1 <= 1e-15 or norm_g_inf <= 1e-15:
                return 0.0

            c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
            return c2
        except Exception:
            return 0.0

class Initializer:
    """Generates diverse initial configurations"""
    
    @staticmethod
    def sophisticated_initialization(n_steps: int = 500) -> List[float]:
        """Generate sophisticated initial configuration"""
        f = np.zeros(n_steps)
        
        segment_size = max(1, n_steps // 12)
        
        for i in range(0, n_steps, segment_size):
            end_idx = min(i + segment_size, n_steps)
            if (i // segment_size) % 2 == 0:
                base_val = 0.75 + np.random.random() * 0.2
                f[i:end_idx] = base_val + np.random.random(end_idx - i) * 0.1
            else:
                f[i:end_idx] = 0.1 + np.random.random(end_idx - i) * 0.15
        
        x = np.linspace(-1, 1, n_steps)
        gaussian = np.exp(-0.5 * (x / 0.22)**2)
        f = f * gaussian * 0.5 + gaussian * 0.5
        
        noise = np.random.normal(0, 0.015, n_steps)
        f = f + noise
        
        f = np.clip(f, 0, None)
        if np.sum(f) > 0:
            f = f / np.sum(f)
        
        return f.tolist()
    
    @staticmethod
    def generate_diverse_population(n_individuals: int, n_steps: int) -> List[List[float]]:
        """Generate diverse initial population"""
        population = []
        
        for i in range(n_individuals):
            if i % 5 == 0:
                f = Initializer._create_alternating_pattern(n_steps)
            elif i % 5 == 1:
                f = Initializer._create_gaussian_pattern(n_steps)
            elif i % 5 == 2:
                f = Initializer._create_uniform_pattern(n_steps)
            elif i % 5 == 3:
                f = Initializer._create_peak_pattern(n_steps)
            else:
                f = Initializer._create_mixed_pattern(n_steps)
            
            population.append(f)
            
        return population
    
    @staticmethod
    def _create_alternating_pattern(n_steps: int) -> List[float]:
        """Create alternating high/low segments"""
        f = np.zeros(n_steps)
        segment_size = max(1, n_steps // 10)
        for j in range(0, n_steps, segment_size):
            end_idx = min(j + segment_size, n_steps)
            if (j // segment_size) % 2 == 0:
                f[j:end_idx] = 0.8 + np.random.random(end_idx - j) * 0.1
            else:
                f[j:end_idx] = 0.1 + np.random.random(end_idx - j) * 0.15
        
        x = np.linspace(-1, 1, n_steps)
        gaussian = np.exp(-0.5 * (x / 0.25)**2)
        f = f * gaussian * 0.6 + gaussian * 0.4
        
        f = np.clip(f, 0, None)
        return f / np.sum(f) if np.sum(f) > 0 else f.tolist()
    
    @staticmethod
    def _create_gaussian_pattern(n_steps: int) -> List[float]:
        """Create Gaussian-like distribution"""
        x = np.linspace(-1, 1, n_steps)
        sigma = 0.15 + np.random.random() * 0.2
        mu = np.random.random() * 0.3 - 0.15
        f = np.exp(-0.5 * ((x - mu) / sigma)**2)
        return f / np.sum(f) if np.sum(f) > 0 else f.tolist()
    
    @staticmethod
    def _create_uniform_pattern(n_steps: int) -> List[float]:
        """Create uniform distribution"""
        f = np.random.random(n_steps)
        f = np.clip(f, 0, 1)
        return f / np.sum(f) if np.sum(f) > 0 else f.tolist()
    
    @staticmethod
    def _create_peak_pattern(n_steps: int) -> List[float]:
        """Create peak-centered distribution"""
        f = np.zeros(n_steps)
        center = n_steps // 2
        width = max(1, n_steps // 12 + np.random.randint(-3, 4))
        f[max(0, center-width//2):min(n_steps, center+width//2)] = 1.0
        f += np.random.normal(0, 0.02, n_steps)
        f = np.clip(f, 0, None)
        return f / np.sum(f) if np.sum(f) > 0 else f.tolist()
    
    @staticmethod
    def _create_mixed_pattern(n_steps: int) -> List[float]:
        """Create mixed multi-peak pattern"""
        f = np.zeros(n_steps)
        peaks = [n_steps // 4, n_steps // 2, 3*n_steps // 4]
        for peak in peaks:
            width = max(1, n_steps // 20)
            start = max(0, peak - width // 2)
            end = min(n_steps, peak + width // 2)
            f[start:end] = 0.6 + np.random.random(end - start) * 0.3
        
        x = np.linspace(-1, 1, n_steps)
        gaussian = np.exp(-0.5 * (x / 0.3)**2)
        f = f * gaussian * 0.5 + gaussian * 0.5
        
        f = np.clip(f, 0, None)
        return f / np.sum(f) if np.sum(f) > 0 else f.tolist()

class EvolutionaryOptimizer:
    """Manages evolutionary optimization pipeline"""
    
    def __init__(self, max_time_seconds: float = 90.0):
        self.max_time_seconds = max_time_seconds
    
    def run_multi_start_optimization(self, n_starts: int = 3) -> List[float]:
        """Run multi-start evolutionary optimization"""
        start_time = time.time()
        n_steps = 500
        best_solution = None
        best_c2 = -np.inf
        
        for start in range(n_starts):
            # Check time limit
            if time.time() - start_time > self.max_time_seconds * 0.9:
                break
                
            try:
                initial_f = Initializer.sophisticated_initialization(n_steps)
                current_c2 = C2ScoreCalculator.calculate_c2(initial_f)
                
                if current_c2 > best_c2:
                    best_c2 = current_c2
                    best_solution = initial_f
                    
                # Run evolutionary optimization
                optimized_f = self._run_single_optimization(n_steps, initial_f, start)
                optimized_c2 = C2ScoreCalculator.calculate_c2(optimized_f)
                
                if optimized_c2 > best_c2:
                    best_c2 = optimized_c2
                    best_solution = optimized_f
                    
            except Exception:
                continue
        
        return best_solution if best_solution is not None else [1.0/n_steps] * n_steps
    
    def _run_single_optimization(self, n_steps: int, initial_f: List[float], start: int) -> List[float]:
        """Run single evolutionary optimization round"""
        bounds = [(0, 1.0) for _ in range(n_steps)]

        def objective(x):
            return -C2ScoreCalculator.calculate_c2(x.tolist())

        # Adaptive parameters
        popsize = 12 if start == 0 else 15
        maxiter = 30 if start == 0 else 20
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                seed=42 + start,
                disp=False
            )
            
            if result.success:
                optimized_f = np.maximum(result.x, 0)
                if np.sum(optimized_f) > 0:
                    optimized_f = optimized_f / np.sum(optimized_f)
                
                # Local refinement
                refined_f = self._local_refinement(optimized_f, bounds, n_steps)
                return refined_f.tolist()
            else:
                return initial_f
                
        except Exception:
            return initial_f
    
    def _local_refinement(self, optimized_f: np.ndarray, bounds: List[Tuple[float, float]], n_steps: int) -> np.ndarray:
        """Apply local refinement with L-BFGS-B"""
        try:
            def local_objective(f_vals):
                return -C2ScoreCalculator.calculate_c2(f_vals.tolist())
            
            local_result = minimize(
                local_objective,
                optimized_f,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 15}
            )
            
            if local_result.success:
                refined_f = np.maximum(local_result.x, 0)
                if np.sum(refined_f) > 0:
                    refined_f = refined_f / np.sum(refined_f)
                return refined_f
        except Exception:
            pass
        
        return optimized_f

def construct_function() -> list[float]:
    """
    Main function to construct step-function with high C2 value
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    try:
        # Initialize components
        evaluator = C2ScoreCalculator()
        initializer = Initializer()
        optimizer = EvolutionaryOptimizer()
        
        # Try sophisticated initialization first
        initial_f = initializer.sophisticated_initialization(500)
        c2_initial = evaluator.calculate_c2(initial_f)
        
        # Run evolutionary optimization
        optimized_f = optimizer.run_multi_start_optimization(3)
        c2_optimized = evaluator.calculate_c2(optimized_f)
        
        # Return better solution
        if c2_optimized > c2_initial:
            return optimized_f
        else:
            return initial_f
            
    except Exception as e:
        # Fallback to simple initialization
        n_steps = 500
        return [1.0/n_steps] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")