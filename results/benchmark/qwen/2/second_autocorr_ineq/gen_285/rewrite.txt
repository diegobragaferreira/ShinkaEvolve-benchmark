# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from typing import List, Tuple
import numba
from numba import jit
from scipy.ndimage import gaussian_filter1d

# Set seeds for reproducibility
np.random.seed(42)

class NormCalculator:
    """Efficient computation of autoconvolution norms using numba acceleration."""
    
    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution_fast(f_vals):
        """Fast autoconvolution computation using Numba."""
        n = len(f_vals)
        g = np.zeros(2 * n - 1)
        
        # Manual convolution for speed
        for i in range(n):
            for j in range(n):
                g[i + j] += f_vals[i] * f_vals[j]
                
        return g
    
    @staticmethod
    @jit(nopython=True)
    def compute_norms_piecewise(g_vals):
        """Compute norms using piecewise linear integration matching evaluator's method."""
        n = len(g_vals)
        
        if n <= 1:
            return 0.0, 0.0, 0.0
            
        # Compute L2 norm squared using trapezoidal-like integration
        # Formula: (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
        norm_2_sq = 0.0
        dx = 0.5 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.5
        
        for i in range(n - 1):
            y1 = g_vals[i]
            y2 = g_vals[i + 1]
            norm_2_sq += (dx / 3.0) * (y1 * y1 + y1 * y2 + y2 * y2)
            
        # Compute L1 norm (sum of absolute values)
        norm_1 = 0.0
        for i in range(n):
            norm_1 += abs(g_vals[i])
            
        # Compute L-infinity norm (maximum absolute value)
        norm_inf = 0.0
        for i in range(n):
            abs_val = abs(g_vals[i])
            if abs_val > norm_inf:
                norm_inf = abs_val
                
        return norm_2_sq, norm_1, norm_inf

class FunctionGenerator:
    """Generates step functions with optimized peak configurations."""
    
    @staticmethod
    def generate_logspaced_peaks(n_points: int, n_peaks: int = None) -> List[Tuple[float, float, float]]:
        """Generate peaks with logarithmic spacing and minimum separation constraints."""
        if n_peaks is None:
            n_peaks = np.random.randint(12, 25)
            
        # Create logarithmically spaced positions avoiding clustering
        peak_positions = []
        
        # Logarithmic distribution from 0.02 to 0.23 (avoiding very close to edges)
        log_min = np.log(0.02)
        log_max = np.log(0.23)
        log_positions = np.logspace(log_min, log_max, n_peaks // 2 + 1)
        
        # Place peaks on both sides with alternating sign
        for i in range(n_peaks):
            side = 1 if i % 2 == 0 else -1
            if i < len(log_positions):
                pos = side * log_positions[i // 2] + np.random.uniform(-0.005, 0.005)
            else:
                # Fill remaining with uniform distribution
                pos = np.random.uniform(-0.23, 0.23)
            
            # Ensure minimum separation from existing peaks
            valid_position = True
            for existing_pos in peak_positions:
                if abs(pos - existing_pos) < 0.015:  # Minimum gap of 0.015
                    valid_position = False
                    break
                    
            if valid_position:
                peak_positions.append(pos)
        
        # Generate peak parameters with adaptive characteristics
        configs = []
        for pos in peak_positions:
            # Distance from center affects parameters 
            center_distance = abs(pos)
            
            # Amplitude: higher near center, lower at edges
            if center_distance < 0.05:
                amp = np.random.uniform(1.5, 2.5)
            elif center_distance < 0.15:
                amp = np.random.uniform(1.0, 2.0)
            else:
                amp = np.random.uniform(0.5, 1.5)
            
            # Width: narrower near center, wider at edges for better convolution profile
            if center_distance < 0.08:
                width = np.random.uniform(0.005, 0.015)
            elif center_distance < 0.18:
                width = np.random.uniform(0.01, 0.03)
            else:
                width = np.random.uniform(0.02, 0.05)
                
            configs.append((pos, amp, width))
            
        return configs
    
    @staticmethod
    def create_function_from_config(configs: List[Tuple[float, float, float]], 
                                  n_points: int) -> List[float]:
        """Create step function from peak configurations with optimized smoothing."""
        x = np.linspace(-0.25, 0.25, n_points)
        f_values = np.zeros_like(x)
        
        # Add all peaks
        for pos, amp, width in configs:
            gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / width)**2)
            f_values += gaussian_peak
            
        # Apply smoothing with adaptive window size
        if n_points > 100:
            # Use adaptive smoothing window based on function complexity
            window_size = max(3, min(21, int(n_points / 50)))
            if window_size % 2 == 0:
                window_size += 1
            try:
                # Try direct Gaussian filtering first
                f_values = gaussian_filter1d(f_values, window_size, mode='nearest')
            except:
                # Fallback to moving average if needed
                f_values = np.convolve(f_values, np.ones(window_size)/window_size, mode='same')
                
        # Ensure non-negativity and normalize
        f_values = np.maximum(f_values, 0)
        max_val = np.max(f_values)
        if max_val > 0:
            f_values = f_values / (max_val * 1.5)  # Slightly less aggressive normalization
            
        return f_values.tolist()

class OptimizerEngine:
    """Core optimization engine with advanced parallel processing capabilities."""
    
    def __init__(self):
        self.norm_calc = NormCalculator()
        self.generator = FunctionGenerator()
        
    def evaluate_single(self, f_values: List[float]) -> float:
        """Evaluate a single candidate function with robust error handling."""
        try:
            # Fast autoconvolution
            f_array = np.array(f_values, dtype=np.float64)
            g = self.norm_calc.compute_autoconvolution_fast(f_array)
            
            # Compute norms
            norm_2_sq, norm_1, norm_inf = self.norm_calc.compute_norms_piecewise(g)
            
            # Avoid division by zero with strict thresholds
            if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                return 0.0
                
            c2 = norm_2_sq / (norm_1 * norm_inf)
            # Additional validation
            if not (np.isnan(c2) or np.isinf(c2) or c2 < 0):
                return c2
            else:
                return 0.0
            
        except Exception as e:
            warnings.warn(f"Evaluation error: {str(e)}")
            return 0.0
    
    def generate_candidate_pool(self, n_candidates: int = 25, n_points: int = 2000) -> List[List[float]]:
        """Generate a pool of candidate functions in parallel with diversified strategies."""
        candidates = []
        
        def create_single_candidate(i):
            try:
                # Use logarithmic spacing with minimum separation
                n_peaks = np.random.randint(15, 30)
                configs = self.generator.generate_logspaced_peaks(n_points, n_peaks)
                f_values = self.generator.create_function_from_config(configs, n_points)
                return f_values
            except Exception as e:
                warnings.warn(f"Candidate {i} generation failed: {str(e)}")
                return None
                
        # Parallel generation with optimized thread usage
        with ThreadPoolExecutor(max_workers=min(12, n_candidates)) as executor:
            futures = [executor.submit(create_single_candidate, i) for i in range(n_candidates)]
            results = [future.result() for future in as_completed(futures)]
            
        # Filter valid candidates
        valid_candidates = [f for f in results if f is not None and len(f) > 10]
        return valid_candidates
    
    def optimize_candidate(self, initial_f: List[float], max_iter: int = 35) -> List[float]:
        """Perform local optimization on a candidate with smart bounds."""
        try:
            def objective(params):
                # Ensure non-negativity
                params = np.maximum(params, 0)
                # Compute norms directly
                g = self.norm_calc.compute_autoconvolution_fast(np.array(params, dtype=np.float64))
                norm_2_sq, norm_1, norm_inf = self.norm_calc.compute_norms_piecewise(g)
                
                # Avoid division by zero
                if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                    return 0.0
                    
                c2 = norm_2_sq / (norm_1 * norm_inf)
                return -c2 if not (np.isnan(c2) or np.isinf(c2)) else 0.0
                
            # Use smarter bounds based on input function characteristics
            bounds = [(0, max(0.1, np.max(initial_f) * 2.0)) for _ in range(len(initial_f))]
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=max_iter,
                popsize=15,  # Slightly larger population
                seed=42,
                strategy='best1bin',
                disp=False
            )
            
            refined_params = np.maximum(result.x, 0).tolist()
            return refined_params
            
        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")
            return initial_f

def construct_function() -> List[float]:
    """Main function to construct step-function with high C2 value."""
    start_time = time.time()
    max_time_seconds = 85
    
    # Initialize optimizer engine
    optimizer = OptimizerEngine()
    
    # Best results tracking
    best_c2 = 0.0
    best_function = []
    
    # Multi-stage optimization with improved strategy
    stages = [
        {"candidates": 20, "points": 1200, "iterations": 15},
        {"candidates": 25, "points": 1800, "iterations": 20}, 
        {"candidates": 15, "points": 2500, "iterations": 25}
    ]
    
    # Track performance for adaptive decisions
    recent_scores = []
    
    for stage_num, stage in enumerate(stages):
        if time.time() - start_time > max_time_seconds - 5:
            break
            
        # Generate candidates for this stage
        candidates = optimizer.generate_candidate_pool(
            n_candidates=stage["candidates"], 
            n_points=stage["points"]
        )
        
        # Evaluate and select best
        for candidate in candidates:
            if time.time() - start_time > max_time_seconds - 5:
                break
            c2 = optimizer.evaluate_single(candidate)
            
            # Track performance
            if not np.isnan(c2) and not np.isinf(c2) and c2 > 0:
                recent_scores.append(c2)
                if len(recent_scores) > 5:
                    recent_scores.pop(0)
            
            if c2 > best_c2:
                best_c2 = c2
                best_function = candidate.copy()
                
        # Early termination if good enough
        if best_c2 > 0.96:
            break
    
    # Final refinement if we have a candidate
    if best_function and time.time() - start_time < max_time_seconds - 5:
        try:
            # Refine with more iterations and better bounds
            refined = optimizer.optimize_candidate(best_function, max_iter=30)
            final_c2 = optimizer.evaluate_single(refined)
            if final_c2 > best_c2 and final_c2 > 0:
                best_c2 = final_c2
                best_function = refined
        except:
            pass
    
    # Final fallback with more robust initialization
    if not best_function:
        n_points = np.random.randint(200, 1000)
        # Create a well-balanced function with both peaks and smooth regions
        f_values = np.zeros(n_points)
        x = np.linspace(-0.25, 0.25, n_points)
        
        # Add a few strategically placed peaks
        n_peaks = max(3, min(8, n_points // 100))
        for _ in range(n_peaks):
            pos = np.random.uniform(-0.20, 0.20)
            amp = np.random.uniform(0.8, 1.8)
            width = np.random.uniform(0.01, 0.04)
            gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / width)**2)
            f_values += gaussian_peak
            
        # Apply smoothing
        if n_points > 100:
            window_size = max(3, min(15, int(n_points / 100)))
            if window_size % 2 == 0:
                window_size += 1
            try:
                f_values = gaussian_filter1d(f_values, window_size, mode='nearest')
            except:
                f_values = np.convolve(f_values, np.ones(window_size)/window_size, mode='same')
                
        f_values = np.maximum(f_values, 0)
        max_val = np.max(f_values)
        if max_val > 0:
            f_values = f_values / (max_val * 1.2)
        
        best_function = f_values.tolist()
        
    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")