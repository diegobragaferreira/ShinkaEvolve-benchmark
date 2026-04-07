# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

class AutoconvolutionEvaluator:
    """Handles all autoconvolution norm computations with numerical stability."""

    @staticmethod
    def compute_norms(f_values):
        """
        Compute the three norms needed for C2 calculation with improved numerical stability.
        Returns: (norm_2_sq, norm_1, norm_inf)
        """
        if not f_values:
            return 0.0, 1e-12, 1e-12

        # Convert to numpy array
        f = np.array(f_values, dtype=np.float64)

        # Validate input
        if len(f) < 1:
            return 0.0, 1e-12, 1e-12

        # Compute autoconvolution using fast convolution
        g = signal.convolve(f, f, mode='full')
        g = g[len(f)-1:]  # Keep only the relevant part

        # Compute norms with numerical stability checks
        g_abs = np.abs(g)

        # ||g||₂² (L2 norm squared) - use trapezoidal rule properly
        g_sq = g_abs ** 2
        if len(g_sq) < 2:
            norm_2_sq = 0.0 if len(g_sq) == 0 else g_sq[0]
        else:
            # Trapezoidal integration: sum((y[i] + y[i+1])/2 * delta_x)
            # We approximate dx as 1 since we're dealing with discrete samples
            norm_2_sq = np.sum((g_sq[:-1] + g_sq[1:]) / 2.0)

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

        # Add top scale peaks
        for i in range(top_scale_peaks):
            gaussian_peak = top_amplitudes[i] * np.exp(-0.5 * ((x - top_positions[i]) / top_widths[i])**2)
            f_values += gaussian_peak

        # Add mid scale peaks
        for i in range(mid_scale_peaks):
            gaussian_peak = mid_amplitudes[i] * np.exp(-0.5 * ((x - mid_positions[i]) / mid_widths[i])**2)
            f_values += gaussian_peak

        # Add bottom scale peaks
        for i in range(bottom_scale_peaks):
            gaussian_peak = bottom_amplitudes[i] * np.exp(-0.5 * ((x - bottom_positions[i]) / bottom_widths[i])**2)
            f_values += gaussian_peak

        # Apply smoothing to reduce sharp edges
        if n_points > 100:
            window_size = max(3, min(25, int(n_points / 40)))
            if window_size % 2 == 0:
                window_size += 1
            try:
                from scipy.signal import savgol_filter
                f_values = savgol_filter(f_values, window_size, 3)
            except:
                # Fallback to simple moving average if savgol fails
                f_values = np.convolve(f_values, np.ones(window_size)/window_size, mode='same')

        # Ensure non-negativity
        f_values = np.maximum(f_values, 0)

        # Normalize for stability
        max_val = np.max(f_values)
        if max_val > 0:
            f_values = f_values / (max_val * 1.8)  # Scale down gently

        return f_values.tolist()

    @staticmethod
    def refine_multiscale_peaks(original_f_values, n_points=2000):
        """
        Refine a multiscale peak function by optimizing peak parameters more carefully.
        This approach focuses on improving peak positions and amplitudes based on
        autoconvolution analysis rather than random generation.
        """
        # Convert to numpy array
        original_f = np.array(original_f_values)

        # Extract peak information from the original function
        # Simple peak detection by finding local maxima
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(original_f)

        if len(peaks) < 2:
            # Not enough peaks, return original
            return original_f_values

        # Sample the function at peak locations to estimate parameters
        # Focus on optimizing a subset of the most significant peaks
        n_refine = min(5, len(peaks))

        # Get peak locations and heights
        peak_positions = []
        peak_heights = []

        # Sample a few peaks to refine
        sample_indices = np.sort(np.random.choice(peaks, min(n_refine, len(peaks)), replace=False))

        for idx in sample_indices:
            peak_positions.append(idx)
            peak_heights.append(original_f[idx])

        # Create new refined function with optimized peak parameters
        x = np.linspace(-0.25, 0.25, n_points)
        refined_f = np.zeros_like(x)

        # Refine based on detected peaks
        # For each peak, try to optimize its location and amplitude to improve autoconvolution
        for i, (pos_idx, height) in enumerate(zip(peak_positions, peak_heights)):
            # Convert index position to actual position in [-0.25, 0.25]
            actual_pos = -0.25 + (pos_idx / (n_points - 1)) * 0.5

            # Determine appropriate width based on neighborhood
            width = 0.01 + 0.02 * np.random.random()  # Reasonable range

            # Adjust height based on context - if a strong peak, make it stronger
            adjusted_height = min(2.0, height * (1.0 + np.random.random() * 0.5))

            # Create refined Gaussian peak
            gaussian_peak = adjusted_height * np.exp(-0.5 * ((x - actual_pos) / width)**2)
            refined_f += gaussian_peak

        # Smooth the result
        if n_points > 100:
            window_size = max(3, min(21, int(n_points / 50)))
            if window_size % 2 == 0:
                window_size += 1
            try:
                refined_f = savgol_filter(refined_f, window_size, 3)
            except:
                refined_f = np.convolve(refined_f, np.ones(window_size)/window_size, mode='same')

        # Ensure non-negativity and normalization
        refined_f = np.maximum(refined_f, 0)
        max_val = np.max(refined_f)
        if max_val > 0:
            refined_f = refined_f / (max_val * 1.5)

        return refined_f.tolist()

    @staticmethod
    def generate_gaussian_peaks(n_points, n_peaks=None):
        """
        Generate step function using Gaussian peaks with logarithmic spacing for better distribution
        """
        if n_peaks is None:
            n_peaks = np.random.randint(5, 26)

        # Create domain
        x = np.linspace(-0.25, 0.25, n_points)
        f_values = np.zeros_like(x)

        # Use logarithmic distribution for peak positions to avoid clustering
        # This helps avoid clustering around center and edges
        peak_positions = []

        # For better distribution, create log-spaced positions
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
            peak_positions.append(pos)

        # Generate peak characteristics with better distribution
        peak_widths = []
        peak_amplitudes = []

        for pos in peak_positions:
            # Vary widths to create more complex profile
            # Wider at center, narrower at edges
            center_distance = abs(pos)
            base_sigma = 0.02 + 0.03 * np.exp(-center_distance * 3.0)
            sigma = np.clip(base_sigma * np.random.uniform(0.7, 1.3), 0.005, 0.08)
            peak_widths.append(sigma)

            # Use distribution that decreases with distance from center
            center_distance = abs(pos)
            # Base amplitude with decay factor based on distance from center
            base_amp = np.random.exponential(0.5) * np.exp(-center_distance * 5.0)
            # Add some variation
            amp = base_amp * np.random.uniform(0.5, 1.5)

            # Cap amplitude to prevent extreme values
            amp = min(1.0, amp)
            peak_amplitudes.append(amp)

        # Add Gaussian peaks
        for i in range(n_peaks):
            # Gaussian peak centered at peak_position with given width and amplitude
            gaussian_peak = peak_amplitudes[i] * np.exp(-0.5 * ((x - peak_positions[i]) / peak_widths[i])**2)
            f_values += gaussian_peak

        # Apply smoothing to reduce sharp edges
        if n_points > 100:
            window_size = max(3, min(21, int(n_points / 50)))
            if window_size % 2 == 0:
                window_size += 1
            try:
                from scipy.signal import savgol_filter
                f_values = savgol_filter(f_values, window_size, 3)
            except:
                # Fallback to simple moving average if savgol fails
                f_values = np.convolve(f_values, np.ones(window_size)/window_size, mode='same')

        # Ensure non-negativity
        f_values = np.maximum(f_values, 0)

        # Normalize for stability
        max_val = np.max(f_values)
        if max_val > 0:
            f_values = f_values / (max_val * 1.5)  # Scale down gently

        return f_values.tolist()

class OptimizerPipeline:
    """Main optimization pipeline that orchestrates function construction and evaluation."""

    def __init__(self):
        self.evaluator = AutoconvolutionEvaluator()
        self.builder = StepFunctionBuilder()

    def evaluate_candidate(self, f_values):
        """Evaluate a single candidate function"""
        try:
            norm_2_sq, norm_1, norm_inf = self.evaluator.compute_norms(f_values)

            # Avoid division by zero
            if norm_1 <= 1e-12 or norm_inf <= 1e-12:
                return 0.0

            c2 = norm_2_sq / (norm_1 * norm_inf)
            return c2 if not np.isnan(c2) and not np.isinf(c2) else 0.0

        except Exception as e:
            warnings.warn(f"Evaluation error: {str(e)}")
            return 0.0

    def construct_candidates_parallel(self, n_candidates=20, n_points=2000):
        """
        Construct multiple candidates in parallel for better exploration
        """
        candidates = []

        def build_single_candidate(index):
            try:
                # Choose construction strategy - now include multiscale approach
                strategy = np.random.choice(['gaussian', 'multiscale'])
                if strategy == 'multiscale':
                    f_values = self.builder.generate_multiscale_peaks(n_points)
                    # Apply refinement to multiscale peaks
                    f_values = self.builder.refine_multiscale_peaks(f_values, n_points)
                else:
                    f_values = self.builder.generate_gaussian_peaks(n_points)
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
                norm_2_sq, norm_1, norm_inf = self.evaluator.compute_norms(params)
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

            # Prioritize multiscale approach first, then gaussian
            strategies = ['multiscale', 'gaussian'] if attempt < 3 else ['gaussian', 'multiscale']

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
                high_res_f = self.builder.generate_gaussian_peaks(5000)
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