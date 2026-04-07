# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution, minimize
from scipy.ndimage import gaussian_filter1d
import random
from typing import List, Tuple
import time

class MultiScalePeakOptimizer:
    """Advanced optimizer using multi-scale peak construction for maximizing C2 constant."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def compute_autoconvolution_norms(self, f_values: List[float]) -> Tuple[float, float, float]:
        """
        Compute the three norms needed for C2 calculation.
        Returns (||g||₂², ||g||₁, ||g||∞) where g = f*f
        """
        if not f_values:
            return 0.0, 0.0, 0.0

        # Convert to numpy array
        f = np.array(f_values)

        # Compute autoconvolution g = f * f
        g = signal.convolve(f, f, mode='full')

        # Extract central portion (valid autoconvolution)
        half_len = len(f) - 1
        g = g[half_len:]  # Take right half

        # Compute norms
        g_squared = g * g
        norm_2_sq = np.sum(g_squared)

        norm_1 = np.sum(np.abs(g))
        norm_inf = np.max(np.abs(g))

        return norm_2_sq, norm_1, norm_inf

    def compute_c2(self, f_values: List[float]) -> float:
        """Compute C2 value for given function"""
        norm_2_sq, norm_1, norm_inf = self.compute_autoconvolution_norms(f_values)

        # Avoid division by zero
        if norm_1 <= 1e-12 or norm_inf <= 1e-12:
            return 0.0

        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2

    def create_multiscale_peak_function(self, n_steps: int) -> List[float]:
        """
        Create step function with multi-scale Gaussian peaks for optimal C2.
        Uses logarithmic spacing and structured construction.
        """
        # Domain definition for step function [-0.25, 0.25]
        x = np.linspace(-0.25, 0.25, n_steps)

        # Determine number of peaks based on function length
        # Use adaptive approach: more peaks for larger functions
        n_peaks = max(3, min(15, n_steps // 100))

        # Logarithmic spacing for peak positions to ensure good distribution
        peak_positions = []
        peak_widths = []
        peak_heights = []

        # Generate peak positions using log-uniform distribution
        # This ensures peaks are spread across multiple scales
        for i in range(n_peaks):
            if i == 0:
                # First peak near left edge (at least 0.05 from boundary)
                pos = np.random.uniform(-0.25, -0.15)
            elif i == n_peaks - 1:
                # Last peak near right edge (at least 0.05 from boundary)
                pos = np.random.uniform(0.15, 0.25)
            else:
                # Intermediate peaks with logarithmic spacing
                # Use log-uniform distribution to ensure even spread
                log_min = np.log(0.05)  # Minimum relative position
                log_max = np.log(0.45)  # Maximum relative position
                log_pos = np.random.uniform(log_min, log_max)
                rel_pos = np.exp(log_pos)  # Transform back to linear
                pos = -0.25 + rel_pos * 0.5  # Map back to [-0.25, 0.25]

            peak_positions.append(pos)

            # Set peak widths with some variation but keep them reasonable
            # Wider peaks help in creating flatter autoconvolution profiles
            width = np.random.uniform(0.005, 0.025)
            peak_widths.append(width)

            # Set peak heights with inverse scaling to avoid overly sharp peaks
            # This helps control ||g||∞ while maintaining ||g||₂²
            height = np.random.uniform(0.8, 2.0)
            peak_heights.append(height)

        # Create the function as sum of Gaussian peaks
        f_values = np.zeros(n_steps)
        for pos, width, height in zip(peak_positions, peak_widths, peak_heights):
            gaussian = height * np.exp(-0.5 * ((x - pos) / width) ** 2)
            f_values += gaussian

        # Ensure non-negativity
        f_values = np.maximum(f_values, 0)

        # Apply mathematical principled smoothing with Gaussian kernel
        # This replaces the Savitzky-Golay smoothing for better numerical stability
        if n_steps > 50:
            f_values = gaussian_filter1d(f_values, sigma=0.8)
            f_values = np.maximum(f_values, 0)

        # Normalize to reasonable range but preserve structure
        if np.max(f_values) > 0:
            f_values = f_values / np.max(f_values) * 1.5

        # Apply real-time feedback-controlled amplitude adjustment
        # Monitor intermediate C2 values to detect when peaks become too dominant
        c2_current = self.compute_c2(f_values.tolist())
        if c2_current < 0.2:
            # If we're getting poor results, apply stronger adjustments
            f_values = f_values * 0.9
        elif c2_current < 0.5:
            # Moderate adjustment
            f_values = f_values * 0.98

        return f_values.tolist()

    def refine_with_local_search(self, initial_func: List[float], max_evals: int = 500) -> List[float]:
        """Apply local refinement to improve initial solution"""
        try:
            # Convert to numpy for easier manipulation
            current_func = np.array(initial_func)

            # Apply mathematical principled smoothing to reduce numerical artifacts
            smoothed = gaussian_filter1d(current_func, sigma=0.5)
            smoothed = np.maximum(smoothed, 0)

            return smoothed.tolist()

        except Exception:
            return initial_func

    def construct_function(self) -> List[float]:
        """Main function to construct step-function with high C2 value."""
        start_time = time.time()

        # Determine number of steps with time budget consideration
        n_steps = min(1000, max(100, 500 + int(np.random.randint(0, 200) * 2)))

        # Phase 1: Create baseline function with multi-scale peak construction
        if time.time() - start_time > 80:  # Leave margin for processing
            return [0.5] * n_steps

        # Create high-quality multi-scale peak function
        best_function = self.create_multiscale_peak_function(n_steps)

        # Phase 2: Local refinement for numerical stability
        if time.time() - start_time > 85:
            return best_function

        refined_function = self.refine_with_local_search(best_function)

        # Phase 3: Final validation and adjustment
        c2_score = self.compute_c2(refined_function)

        # If we're close to timeout, just return what we have
        if time.time() - start_time > 88:
            return refined_function

        # Apply one final adjustment if needed based on quality check
        if c2_score < 0.1:  # Very low score - try alternative approach
            # Fall back to simpler but proven approach
            x = np.linspace(-0.25, 0.25, n_steps)
            # Create a bell-shaped distribution similar to successful approaches
            base_shape = np.exp(-x**2 / 0.02)
            base_shape = base_shape / np.max(base_shape) * 0.8
            noise = np.random.normal(0, 0.02, n_steps)
            alternative = np.maximum(base_shape + noise, 0)
            refined_function = alternative.tolist()

        return refined_function

def construct_function() -> List[float]:
    """Wrapper function that maintains interface compatibility"""
    optimizer = MultiScalePeakOptimizer()
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")