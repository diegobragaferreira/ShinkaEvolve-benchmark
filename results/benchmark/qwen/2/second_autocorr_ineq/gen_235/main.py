# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
from scipy.ndimage import gaussian_filter1d
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import numba
from numba import jit
import math

@jit(nopython=True)
def fast_trapezoidal_integration(y_vals):
    """Fast trapezoidal integration for numba compatibility"""
    if len(y_vals) < 2:
        return 0.0 if len(y_vals) == 0 else y_vals[0]
    
    integral = 0.0
    for i in range(len(y_vals) - 1):
        integral += (y_vals[i] + y_vals[i+1]) / 2.0
    return integral

class AutoconvolutionEvaluator:
    """Handles all autoconvolution norm computations with numerical stability and speed."""

    @staticmethod
    def compute_norms_fast(f_values):
        """
        Fast computation of norms for C2 calculation with minimal overhead.
        Returns: (norm_2_sq, norm_1, norm_inf)
        """
        if not f_values:
            return 0.0, 1e-12, 1e-12

        # Convert to numpy array
        f = np.array(f_values, dtype=np.float64)

        # Validate input
        if len(f) < 1:
            return 0.0, 1e-12, 1e-12

        # Use FFT-based convolution for speed
        # Pad array to avoid circular convolution effects
        n = len(f)
        padded_len = 2 * n - 1
        f_padded = np.pad(f, (0, padded_len - n), 'constant', constant_values=0)
        
        # FFT convolution
        f_fft = np.fft.fft(f_padded)
        g_fft = f_fft * np.conj(f_fft)
        g = np.fft.ifft(g_fft).real[:padded_len]
        
        # Keep only the middle part (proper autoconvolution)
        g = g[n-1:2*n-1]
        
        # Compute norms efficiently
        g_abs = np.abs(g)
        
        # ||g||₂² using fast trapezoidal integration
        g_sq = g_abs ** 2
        norm_2_sq = fast_trapezoidal_integration(g_sq)

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
    def generate_optimized_gaussian_peaks(n_points, n_peaks=None):
        """
        Generate step function using optimized Gaussian peaks with enhanced distribution
        """
        if n_peaks is None:
            n_peaks = np.random.randint(12, 35)

        # Create domain
        x = np.linspace(-0.25, 0.25, n_points)
        f_values = np.zeros_like(x)

        # Use logarithmic distribution for peak positions with better control
        peak_positions = []
        peak_amplitudes = []
        peak_widths = []

        # Create log-spaced positions with careful spacing
        log_min = np.log(0.015)
        log_max = np.log(0.15)
        n_log_peaks = min(n_peaks, 20)
        log_spaced_positions = np.logspace(log_min, log_max, n_log_peaks)

        # Place peaks on both sides of center with logarithmic spacing
        placed_positions = set()
        for i in range(n_peaks):
            if i < len(log_spaced_positions):
                # Alternate sides to distribute evenly
                side = 1 if i % 2 == 0 else -1
                pos = side * log_spaced_positions[i // 2] + np.random.uniform(-0.008, 0.008)
            else:
                # Fill remaining peaks with uniform distribution
                pos = np.random.uniform(-0.23, 0.23)

            # Clip and ensure uniqueness
            pos = np.clip(pos, -0.23, 0.23)
            
            # Check for proximity to existing positions
            valid_position = True
            for existing_pos in placed_positions:
                if abs(pos - existing_pos) < 0.015:
                    valid_position = False
                    break
                    
            if valid_position:
                peak_positions.append(pos)
                placed_positions.add(pos)

        # Generate peak characteristics with better distribution
        for pos in peak_positions:
            # Base amplitude with distance-dependent decay
            center_distance = abs(pos)
            base_amp = np.random.exponential(0.7) * np.exp(-center_distance * 4.0)
            amp = min(1.0, base_amp * np.random.uniform(0.6, 1.4))
            peak_amplitudes.append(amp)

        # Create Gaussian peaks with optimized widths
        for i, (pos, amp) in enumerate(zip(peak_positions, peak_amplitudes)):
            # Width varies with position - wider at edges, narrower at center
            center_distance = abs(pos)
            base_sigma = 0.02 + 0.04 * np.exp(-center_distance * 2.0)
            sigma = np.clip(base_sigma * np.random.uniform(0.6, 1.4), 0.008, 0.08)
            peak_widths.append(sigma)

            gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
            f_values += gaussian_peak

        # Apply smoothing with optimized parameters
        if n_points > 100:
            window_size = max(3, min(25, int(n_points / 30)))
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
        self.builder = StepFunctionBuilder()

    def evaluate_candidate(self, f_values):
        """Evaluate a single candidate function with fast evaluation"""
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

    def construct_candidates_parallel(self, n_candidates=30, n_points=2000):
        """
        Construct multiple candidates in parallel for better exploration
        """
        candidates = []

        def build_single_candidate(index):
            try:
                # Focus on optimized Gaussian approach
                f_values = self.builder.generate_optimized_gaussian_peaks(n_points)
                c2 = self.evaluate_candidate(f_values)
                return (c2, f_values)
            except Exception as e:
                warnings.warn(f"Candidate {index} construction failed: {str(e)}")
                return (0.0, [])

        # Process candidates in parallel with optimized thread count
        max_workers = min(12, n_candidates)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(build_single_candidate, i) for i in range(n_candidates)]
            results = [future.result() for future in as_completed(futures)]

        # Filter out invalid candidates
        valid_candidates = [(c2, f_values) for c2, f_values in results if f_values and len(f_values) > 0]

        # Sort by C2 score
        valid_candidates.sort(key=lambda x: x[0], reverse=True)

        return valid_candidates

    def refine_best_candidate(self, best_f, max_iterations=25):
        """
        Perform local refinement on the best candidate with faster optimization
        """
        try:
            # Simplified refinement using direct parameter adjustments for speed
            def objective(params):
                params = np.maximum(params, 0)
                norm_2_sq, norm_1, norm_inf = self.evaluator.compute_norms_fast(params)
                if norm_1 <= 1e-12 or norm_inf <= 1e-12:
                    return 0.0
                c2 = norm_2_sq / (norm_1 * norm_inf)
                return -c2 if not np.isnan(c2) and not np.isinf(c2) else 0.0

            # Use a more efficient optimization approach
            bounds = [(0, 1) for _ in range(len(best_f))]
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=max_iterations,
                popsize=8,
                seed=42,
                strategy='best1bin',
                tol=0.01,
                recombination=0.7
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

        # Try multiple candidate generations with focused approach
        for attempt in range(6):  # More iterations to escape local minima
            if time.time() - start_time > max_time_seconds:
                break

            # Generate candidates in parallel with focus on optimized approach
            candidates = self.construct_candidates_parallel(
                n_candidates=25, 
                n_points=1800
            )

            # Select best among generated candidates
            for c2, f_values in candidates:
                if c2 > best_c2:
                    best_c2 = c2
                    best_f = f_values

            # Early termination if we have a strong result
            if best_c2 > 0.96:
                break

        # Final refinement with increased resolution if time permits
        if best_f is not None and time.time() - start_time < max_time_seconds * 0.85:
            try:
                # High resolution refinement
                high_res_f = self.builder.generate_optimized_gaussian_peaks(4000)
                high_res_c2 = self.evaluate_candidate(high_res_f)

                if high_res_c2 > best_c2:
                    best_c2 = high_res_c2
                    best_f = high_res_f

                # Local optimization with limited iterations
                best_f = self.refine_best_candidate(best_f, max_iterations=15)

            except Exception as e:
                warnings.warn(f"Final refinement failed: {str(e)}")

        # Fallback if nothing worked
        if best_f is None:
            # Basic fallback with consistent parameters
            n_points = 800
            best_f = [np.random.random() * 0.6 for _ in range(n_points)]

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