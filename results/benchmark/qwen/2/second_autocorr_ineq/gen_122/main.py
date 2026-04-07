# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
from scipy.ndimage import gaussian_filter1d
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import math

class MultiScaleGaussianOptimizer:
    """Advanced multi-scale Gaussian function optimizer for maximizing C2 constant."""

    def __init__(self):
        self.max_evaluations = 1000

    def compute_autoconvolution_norms(self, f_values):
        """Compute the norms needed for C2 calculation with enhanced numerical stability."""
        if not f_values or len(f_values) < 1:
            return 0.0, 1e-12, 1e-12

        # Convert to numpy array
        f = np.array(f_values, dtype=np.float64)

        # Compute autoconvolution
        g = signal.convolve(f, f, mode='full')
        g = g[len(f)-1:]  # Keep only the relevant part

        # Compute norms with numerical stability checks
        g_abs = np.abs(g)

        # ||g||₂² (L2 norm squared) - trapezoidal rule
        g_sq = g_abs ** 2
        if len(g_sq) < 2:
            norm_2_sq = 0.0 if len(g_sq) == 0 else g_sq[0]
        else:
            # Proper trapezoidal integration for discrete samples
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

    def generate_multiscale_gaussian(self, n_points, min_peaks=5, max_peaks=30):
        """Generate function using multi-scale Gaussian components."""
        np.random.seed(42)

        # Create domain
        x = np.linspace(-0.25, 0.25, n_points)
        f_values = np.zeros_like(x)

        # Determine number of peaks
        n_peaks = np.random.randint(min_peaks, max_peaks + 1)

        # Generate peaks at multiple scales with logarithmic spacing
        peak_positions = []
        peak_amplitudes = []
        peak_widths = []

        # Use logarithmic spacing for better distribution
        # This ensures peaks are spread across the domain without clustering
        log_min = np.log(0.015)  # Minimum distance from center
        log_max = np.log(0.23)   # Maximum distance from center
        log_spaced_positions = np.logspace(log_min, log_max, n_peaks // 2 + 1)

        # Place peaks with logarithmic distribution
        # Alternate between positive and negative sides for better symmetry
        for i in range(n_peaks):
            if i < len(log_spaced_positions):
                pos = (-1)**i * log_spaced_positions[i // 2]
                # Add small random perturbation to avoid perfect symmetry
                pos += np.random.normal(0, 0.005)
            else:
                # For additional peaks, place randomly but ensure they're not too close to center
                pos = np.random.uniform(-0.23, 0.23)
                # Ensure minimum distance from center
                if abs(pos) < 0.03:
                    pos = np.sign(pos) * (0.03 + np.random.uniform(0, 0.05))

            # Clip to valid range
            pos = np.clip(pos, -0.25, 0.25)
            peak_positions.append(pos)

        # Remove duplicate positions and ensure enough unique peaks
        unique_positions = []
        for pos in peak_positions:
            # Check if position is sufficiently far from others (minimum 0.02 spacing)
            is_valid = True
            for existing in unique_positions:
                if abs(pos - existing) < 0.02:  # Enforced minimum spacing constraint
                    is_valid = False
                    break
            if is_valid:
                unique_positions.append(pos)

        # If we don't have enough unique positions, add more
        while len(unique_positions) < n_peaks:
            pos = np.random.uniform(-0.23, 0.23)
            # Ensure minimum distance from existing positions
            is_valid = True
            for existing in unique_positions:
                if abs(pos - existing) < 0.02:
                    is_valid = False
                    break
            if is_valid:
                unique_positions.append(pos)

        # Limit to exact number of requested peaks
        unique_positions = unique_positions[:n_peaks]

        # Generate all peak parameters with better distribution
        for pos in unique_positions:
            # Amplitude with distance-dependent decay
            center_distance = abs(pos)
            # Use more controlled amplitude decay based on distance from center
            base_amp = 0.5 + 0.5 * np.exp(-center_distance * 3.0)  # Changed decay factor
            amp = base_amp * np.random.uniform(0.8, 1.2)  # Reduced variation
            amp = min(1.0, amp)  # Cap amplitude
            peak_amplitudes.append(amp)

            # Width with distance-dependent scaling - more controlled variation
            base_width = 0.015 + 0.035 * (1.0 - np.exp(-center_distance * 1.5))
            width = np.clip(base_width * np.random.uniform(0.9, 1.1), 0.005, 0.08)
            peak_widths.append(width)

        # Build the function with multi-scale approach
        for i, (pos, amp, width) in enumerate(zip(unique_positions, peak_amplitudes, peak_widths)):
            gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / width) ** 2)
            f_values += gaussian_peak

        # Apply structured smoothing to reduce sharp edges but maintain structure
        if n_points > 100:
            # Adaptive smoothing based on function complexity
            smooth_width = max(1, min(15, int(n_points * 0.02)))
            if smooth_width % 2 == 0:
                smooth_width += 1
            f_values = gaussian_filter1d(f_values, smooth_width, mode='nearest')

        # Ensure non-negativity and normalize
        f_values = np.maximum(f_values, 0)
        max_val = np.max(f_values)
        if max_val > 0:
            f_values = f_values / (max_val * 1.3)  # Gentle normalization

        return f_values.tolist()

    def coordinate_ascent_refinement(self, initial_f, max_iter=20):
        """Refine function using coordinate ascent approach."""
        f_values = np.array(initial_f, dtype=np.float64)
        n_points = len(f_values)

        # Simple refinement by adjusting peak amplitudes and positions
        # This focuses on improving the autoconvolution structure
        for iteration in range(max_iter):
            try:
                # Compute current C2
                norm_2_sq, norm_1, norm_inf = self.compute_autoconvolution_norms(f_values)
                if norm_1 <= 1e-12 or norm_inf <= 1e-12:
                    break

                current_c2 = norm_2_sq / (norm_1 * norm_inf)

                # Slightly modify function to see improvement
                # Create new candidate by making small adjustments
                candidate = f_values.copy()

                # Adjust a few randomly selected points
                indices_to_adjust = np.random.choice(n_points, size=min(10, n_points//4), replace=False)
                for idx in indices_to_adjust:
                    # Small perturbation
                    perturbation = np.random.normal(0, 0.01 * np.std(candidate))
                    new_value = candidate[idx] + perturbation
                    candidate[idx] = max(0, new_value)  # Ensure non-negativity

                # Test the candidate
                norm_2_sq_new, norm_1_new, norm_inf_new = self.compute_autoconvolution_norms(candidate)
                if norm_1_new <= 1e-12 or norm_inf_new <= 1e-12:
                    continue

                new_c2 = norm_2_sq_new / (norm_1_new * norm_inf_new)

                # Accept improvement
                if new_c2 > current_c2:
                    f_values = candidate

            except Exception as e:
                warnings.warn(f"Coordinate ascent refinement error: {str(e)}")
                break

        return f_values.tolist()

    def construct_optimized_function(self, max_time_seconds=85):
        """Main function construction with multi-scale optimization."""
        start_time = time.time()
        best_c2 = 0.0
        best_f = None

        # Phase 1: Multiple candidate generation with different strategies
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            for strategy_id in range(8):  # Multiple strategies
                if time.time() - start_time > max_time_seconds * 0.8:
                    break

                def worker_func(strategy):
                    try:
                        n_points = 2000 + (strategy * 100)  # Vary resolution
                        # Strategy-specific parameters
                        if strategy % 3 == 0:
                            # High-resolution with many peaks
                            f_values = self.generate_multiscale_gaussian(n_points, 15, 35)
                        elif strategy % 3 == 1:
                            # Medium resolution with moderate peaks
                            f_values = self.generate_multiscale_gaussian(n_points, 8, 20)
                        else:
                            # Low resolution with fewer peaks
                            f_values = self.generate_multiscale_gaussian(n_points, 5, 12)

                        c2 = self.compute_autoconvolution_norms(f_values)[0] / (
                            self.compute_autoconvolution_norms(f_values)[1] *
                            self.compute_autoconvolution_norms(f_values)[2] + 1e-15
                        )

                        return (c2, f_values)
                    except Exception as e:
                        warnings.warn(f"Worker strategy {strategy} failed: {str(e)}")
                        return (0.0, [])

                futures.append(executor.submit(worker_func, strategy_id))

            # Collect results
            for future in as_completed(futures):
                try:
                    c2, f_values = future.result()
                    if c2 > best_c2:
                        best_c2 = c2
                        best_f = f_values
                except Exception as e:
                    warnings.warn(f"Future result error: {str(e)}")

        # Phase 2: Refinement of best candidate
        if best_f is not None and time.time() - start_time < max_time_seconds * 0.9:
            try:
                # Multi-stage refinement
                refined_f = best_f

                # Stage 1: Coordinate ascent
                refined_f = self.coordinate_ascent_refinement(refined_f, 15)

                # Stage 2: Local optimization if time allows
                if time.time() - start_time < max_time_seconds * 0.95:
                    # Final optimization using differential evolution with reduced bounds
                    def objective(params):
                        norm_2_sq, norm_1, norm_inf = self.compute_autoconvolution_norms(params)
                        if norm_1 <= 1e-12 or norm_inf <= 1e-12:
                            return 0.0
                        c2 = norm_2_sq / (norm_1 * norm_inf)
                        return -c2 if not np.isnan(c2) and not np.isinf(c2) else 0.0

                    bounds = [(0, 1) for _ in range(len(best_f))]
                    result = differential_evolution(
                        objective,
                        bounds=bounds,
                        maxiter=20,  # Reduced iterations due to time constraint
                        popsize=5,   # Reduced population
                        seed=42,
                        strategy='best1bin'
                    )

                    refined_params = result.x
                    refined_params = np.maximum(refined_params, 0)
                    refined_c2 = self.compute_autoconvolution_norms(refined_params)[0] / (
                        self.compute_autoconvolution_norms(refined_params)[1] *
                        self.compute_autoconvolution_norms(refined_params)[2] + 1e-15
                    )

                    if refined_c2 > best_c2:
                        best_c2 = refined_c2
                        refined_f = refined_params.tolist()

                best_f = refined_f

            except Exception as e:
                warnings.warn(f"Refinement failed: {str(e)}")
                pass

        # Phase 3: Fallback if nothing worked well
        if best_f is None:
            # Fallback to sophisticated structured approach
            n_points = 1000
            x = np.linspace(-0.25, 0.25, n_points)
            f_values = np.zeros(n_points)

            # Create a multi-scale approach with known good characteristics
            # Use a combination of different scales
            scales = [0.02, 0.03, 0.04, 0.05, 0.06]
            n_peaks_per_scale = 3

            for i, scale in enumerate(scales):
                n_peaks = n_peaks_per_scale
                positions = np.linspace(-0.23, 0.23, n_peaks)
                if i % 2 == 0:  # Add some randomness
                    positions += np.random.normal(0, 0.005, len(positions))

                for pos in positions:
                    if -0.25 <= pos <= 0.25:
                        width = scale * (1.0 + np.random.uniform(-0.1, 0.1))
                        height = 0.5 + 0.3 * np.random.uniform(0, 1)
                        peak = height * np.exp(-0.5 * ((x - pos) / width)**2)
                        f_values += peak

            # Final smoothing and normalization
            f_values = np.maximum(f_values, 0)
            max_val = np.max(f_values)
            if max_val > 0:
                f_values = f_values / (max_val * 1.2)

            best_f = f_values.tolist()

        return best_f

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    optimizer = MultiScaleGaussianOptimizer()
    return optimizer.construct_optimized_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")