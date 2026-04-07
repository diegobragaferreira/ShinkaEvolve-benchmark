# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
from scipy.ndimage import gaussian_filter1d
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from numba import jit, prange

@jit(nopython=True)
def compute_trapezoidal_norm_2_sq(g_values):
    """Fast computation of ||g||₂² using trapezoidal rule"""
    if len(g_values) <= 1:
        return g_values[0]**2 if len(g_values) > 0 else 0.0
    
    norm_2_sq = 0.0
    for i in range(len(g_values)-1):
        y1, y2 = g_values[i], g_values[i+1]
        norm_2_sq += (1.0/3.0) * (y1**2 + y1*y2 + y2**2)
    return norm_2_sq

class AutoconvolutionEvaluator:
    """Handles all autoconvolution norm computations with numerical stability and performance."""
    
    @staticmethod
    @jit(nopython=True)
    def compute_norms_fast(f_values):
        """
        Fast computation of the three norms needed for C2 calculation.
        Uses numba-optimized implementation.
        """
        if len(f_values) == 0:
            return 0.0, 1e-12, 1e-12
            
        # Convert to numpy array
        f = np.array(f_values, dtype=np.float64)
        
        # Compute autoconvolution using numpy (already optimized)
        g = signal.convolve(f, f, mode='full')
        g = g[len(f)-1:]  # Keep only the relevant part
        
        # Compute norms with numba optimization
        g_abs = np.abs(g)
        
        # Fast computation of ||g||₂² 
        norm_2_sq = compute_trapezoidal_norm_2_sq(g_abs)
        
        # ||g||₁ (L1 norm) - sum of absolute values
        norm_1 = np.sum(g_abs)
        
        # ||g||∞ (infinity norm)
        norm_inf = np.max(g_abs)
        
        # Numerical stability checks
        norm_2_sq = max(0.0, norm_2_sq)
        norm_1 = max(1e-12, norm_1)  # Avoid division by zero
        norm_inf = max(1e-12, norm_inf)  # Avoid division by zero
        
        return norm_2_sq, norm_1, norm_inf

class StepFunctionBuilder:
    """Handles construction of step functions with various strategies."""
    
    @staticmethod
    def generate_multiscale_peaks(n_points, n_peaks=None):
        """
        Generate step function using multi-scale peak construction for improved optimization
        """
        if n_peaks is None:
            n_peaks = np.random.randint(10, 35)

        # Create domain
        x = np.linspace(-0.25, 0.25, n_points)
        f_values = np.zeros_like(x)

        # Multi-scale approach: Different regions get different peak characteristics
        # Top scale: dominant peaks near center
        top_scale_peaks = max(3, min(8, n_peaks // 3))
        mid_scale_peaks = max(3, min(12, n_peaks // 2))
        bottom_scale_peaks = n_peaks - top_scale_peaks - mid_scale_peaks

        # Top scale: High amplitude, narrow peaks near center
        top_positions = np.random.uniform(-0.08, 0.08, top_scale_peaks)
        top_amplitudes = np.random.uniform(1.2, 2.0, top_scale_peaks)
        top_widths = np.random.uniform(0.005, 0.02, top_scale_peaks)

        # Mid scale: Medium amplitude, medium width
        mid_positions = np.random.uniform(-0.15, 0.15, mid_scale_peaks)
        mid_amplitudes = np.random.uniform(0.8, 1.5, mid_scale_peaks)
        mid_widths = np.random.uniform(0.015, 0.04, mid_scale_peaks)

        # Bottom scale: Low amplitude, wide peaks near edges
        bottom_positions = np.random.uniform(-0.23, 0.23, bottom_scale_peaks)
        bottom_amplitudes = np.random.uniform(0.3, 0.8, bottom_scale_peaks)
        bottom_widths = np.random.uniform(0.03, 0.07, bottom_scale_peaks)

        # Add peaks using vectorized operations for speed
        for i in range(top_scale_peaks):
            gaussian_peak = top_amplitudes[i] * np.exp(-0.5 * ((x - top_positions[i]) / top_widths[i])**2)
            f_values += gaussian_peak

        for i in range(mid_scale_peaks):
            gaussian_peak = mid_amplitudes[i] * np.exp(-0.5 * ((x - mid_positions[i]) / mid_widths[i])**2)
            f_values += gaussian_peak

        for i in range(bottom_scale_peaks):
            gaussian_peak = bottom_amplitudes[i] * np.exp(-0.5 * ((x - bottom_positions[i]) / bottom_widths[i])**2)
            f_values += gaussian_peak

        # Apply smoothing using numba-optimized gaussian filter
        if n_points > 100:
            window_size = max(3, min(25, int(n_points / 40)))
            if window_size % 2 == 0:
                window_size += 1
            try:
                f_values = gaussian_filter1d(f_values, window_size, mode='nearest')
            except:
                f_values = np.convolve(f_values, np.ones(window_size)/window_size, mode='same')

        # Ensure non-negativity
        f_values = np.maximum(f_values, 0)

        # Normalize for stability
        max_val = np.max(f_values)
        if max_val > 0:
            f_values = f_values / (max_val * 1.8)

        return f_values.tolist()

    @staticmethod
    def generate_adaptive_gaussian_peaks(n_points, n_peaks=None):
        """
        Generate step function using adaptive Gaussian peaks with better distribution
        """
        if n_peaks is None:
            n_peaks = np.random.randint(8, 30)

        # Create domain
        x = np.linspace(-0.25, 0.25, n_points)
        f_values = np.zeros_like(x)

        # Use logarithmic distribution for peak positions to avoid clustering
        peak_positions = []
        peak_amplitudes = []
        peak_widths = []

        # Logarithmic distribution of peak positions
        log_min = np.log(0.02)
        log_max = np.log(0.12)
        log_spaced_positions = np.logspace(log_min, log_max, n_peaks // 2 + 1)

        # Place peaks on both sides of center with logarithmic spacing
        for i in range(n_peaks):
            # Alternate sides to distribute evenly
            side = 1 if i % 2 == 0 else -1
            if i < len(log_spaced_positions):
                pos = side * log_spaced_positions[i // 2] + np.random.uniform(-0.01, 0.01)
            else:
                # Fill remaining peaks with uniform distribution
                pos = np.random.uniform(-0.23, 0.23)

            # Clip to valid range
            pos = np.clip(pos, -0.23, 0.23)

            # Avoid too close proximity
            valid_position = True
            for existing_pos in peak_positions:
                if abs(pos - existing_pos) < 0.02:  # Minimum gap of 0.02
                    valid_position = False
                    break

            if valid_position:
                peak_positions.append(pos)

        # Generate amplitudes with better distribution using vectorized approach
        pos_array = np.array(peak_positions)
        center_distances = np.abs(pos_array)
        
        # Vectorized amplitude generation
        base_amps = np.random.exponential(0.5, len(peak_positions)) * np.exp(-center_distances * 5.0)
        amp_factors = np.random.uniform(0.5, 1.5, len(peak_positions))
        peak_amplitudes = np.minimum(1.0, base_amps * amp_factors)

        # Create Gaussian peaks with optimized widths using vectorized operations
        center_distances = np.abs(np.array(peak_positions))
        base_sigmas = 0.02 + 0.03 * np.exp(-center_distances * 3.0)
        sigma_factors = np.random.uniform(0.7, 1.3, len(peak_positions))
        peak_widths = np.clip(base_sigmas * sigma_factors, 0.005, 0.08)
        
        # Vectorized Gaussian peak addition
        for i, (pos, amp, sigma) in enumerate(zip(peak_positions, peak_amplitudes, peak_widths)):
            gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
            f_values += gaussian_peak

        # Apply smoothing to reduce sharp edges
        if n_points > 100:
            window_size = max(3, min(21, int(n_points / 50)))
            if window_size % 2 == 0:
                window_size += 1
            try:
                f_values = gaussian_filter1d(f_values, window_size, mode='nearest')
            except:
                # Fallback to simple moving average if gaussian_filter fails
                f_values = np.convolve(f_values, np.ones(window_size)/window_size, mode='same')

        # Ensure non-negativity
        f_values = np.maximum(f_values, 0)

        # Normalize for stability
        max_val = np.max(f_values)
        if max_val > 0:
            f_values = f_values / (max_val * 1.5)

        return f_values.tolist()

class OptimizerPipeline:
    """Main optimization pipeline that orchestrates function construction and evaluation."""
    
    def __init__(self):
        self.evaluator = AutoconvolutionEvaluator()
        
    def evaluate_candidate(self, f_values):
        """Evaluate a single candidate function"""
        try:
            norm_2_sq, norm_1, norm_inf = self.evaluator.compute_norms_fast(f_values)
            
            # Avoid division by zero
            if norm_1 <= 1e-12 or norm_inf <= 1e-12:
                return 0.0
                
            c2 = norm_2_sq / (norm_1 * norm_inf)
            return c2 if not np.isnan(c2) and not np.isinf(c2) else 0.0
            
        except Exception as e:
            warnings.warn(f"Evaluation error: {str(e)}")
            return 0.0
            
    def construct_candidates_parallel(self, n_candidates=20, n_points=2000, strategy='adaptive'):
        """
        Construct multiple candidates in parallel for better exploration
        """
        candidates = []
        
        def build_single_candidate(index):
            try:
                # Choose construction strategy - now include multiscale approach
                if np.random.random() < 0.5:
                    strategy = 'multiscale'
                else:
                    strategy = 'adaptive'
                    
                builder = StepFunctionBuilder()
                if strategy == 'multiscale':
                    f_values = builder.generate_multiscale_peaks(n_points)
                else:
                    f_values = builder.generate_adaptive_gaussian_peaks(n_points)
                    
                c2 = self.evaluate_candidate(f_values)
                return (c2, f_values)
            except Exception as e:
                warnings.warn(f"Candidate {index} construction failed: {str(e)}")
                return (0.0, [])
        
        # Process candidates in parallel
        with ThreadPoolExecutor(max_workers=min(8, n_candidates)) as executor:
            futures = [executor.submit(build_single_candidate, i) for i in range(n_candidates)]
            results = [future.result() for future in as_completed(futures)]
            
        # Filter out invalid candidates
        valid_candidates = [(c2, f_values) for c2, f_values in results if f_values and len(f_values) > 0]
        
        # Sort by C2 score
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        
        return valid_candidates
    
    def refine_best_candidate(self, best_f, max_iterations=30):
        """
        Perform local refinement on the best candidate
        """
        try:
            # Simple gradient-free refinement using differential evolution
            def objective(params):
                # Normalize and ensure non-negativity
                params = np.maximum(params, 0)
                # For simplicity, we're just evaluating the function
                norm_2_sq, norm_1, norm_inf = self.evaluator.compute_norms_fast(params)
                if norm_1 <= 1e-12 or norm_inf <= 1e-12:
                    return 0.0
                c2 = norm_2_sq / (norm_1 * norm_inf)
                return -c2 if not np.isnan(c2) and not np.isinf(c2) else 0.0
            
            # Run differential evolution for refinement
            bounds = [(0, 1) for _ in range(len(best_f))]
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=max_iterations,
                popsize=10,
                seed=42,
                strategy='best1bin'
            )
            
            # Return refined result if it's better
            refined_params = result.x
            refined_params = np.maximum(refined_params, 0)
            refined_c2 = self.evaluate_candidate(refined_params)
            
            if refined_c2 > self.evaluate_candidate(best_f):
                return refined_params.tolist()
                
        except Exception as e:
            warnings.warn(f"Refinement failed: {str(e)}")
            
        return best_f
    
    def construct_optimized_function(self, max_time_seconds=85):
        """
        Main function construction routine with optimized pipeline
        """
        start_time = time.time()
        best_c2 = 0.0
        best_f = None
        
        # Try multiple candidate generations
        for attempt in range(5):  # Multiple passes to avoid local optima
            if time.time() - start_time > max_time_seconds:
                break
                
            # Generate candidates in parallel with mixed strategies
            candidates = self.construct_candidates_parallel(n_candidates=15, n_points=1500)
            
            # Select best among generated candidates
            for c2, f_values in candidates:
                if c2 > best_c2:
                    best_c2 = c2
                    best_f = f_values
                    
            # Early termination if we already have good result
            if best_c2 > 0.95:
                break
                
        # Final refinement of the best candidate
        if best_f is not None and time.time() - start_time < max_time_seconds * 0.9:
            try:
                # Increase resolution for final refinement
                builder = StepFunctionBuilder()
                high_res_f = builder.generate_adaptive_gaussian_peaks(5000)
                high_res_c2 = self.evaluate_candidate(high_res_f)
                
                if high_res_c2 > best_c2:
                    best_c2 = high_res_c2
                    best_f = high_res_f
                    
                # Local optimization on the best found function
                best_f = self.refine_best_candidate(best_f, max_iterations=20)
                
            except Exception as e:
                warnings.warn(f"Final refinement failed: {str(e)}")
        
        # Fallback if nothing worked
        if best_f is None:
            # Basic fallback - uniform random distribution
            n_points = np.random.randint(100, 500)
            best_f = [np.random.random() * 0.5 for _ in range(n_points)]
            
        return best_f

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Create pipeline instance and run optimization
    pipeline = OptimizerPipeline()
    return pipeline.construct_optimized_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")