# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution, minimize
from scipy.ndimage import gaussian_filter1d
import random
from typing import List, Tuple
import time

class AutoCorrelationOptimizer:
    """Main optimizer class for maximizing C2 constant through step function construction."""

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

    def create_initial_function(self, n_steps: int) -> List[float]:
        """Create initial candidate function with enhanced peak distribution for better C2"""
        # Create a multi-peak structure with strategic positioning
        x = np.linspace(-0.25, 0.25, n_steps)
        f_values = np.zeros(n_steps)

        # Use logarithmic spacing for better peak distribution
        n_peaks = min(8, max(3, n_steps // 150))

        # Generate logarithmically spaced peak positions
        # This helps avoid clustering and ensures good distribution across the domain
        log_min = np.log(0.02)  # Avoid very close to edges
        log_max = np.log(0.48)  # Avoid very close to edges
        log_positions = np.linspace(log_min, log_max, n_peaks)
        peak_positions = -0.25 + 0.5 * (1 + np.exp(log_positions) / (1 + np.exp(log_positions)) * 0.48)

        # Apply more sophisticated peak characteristics
        peak_heights = []
        peak_widths = []

        for i, pos in enumerate(peak_positions):
            # Heights based on position to avoid overly dominant peaks at edges
            # Peaks closer to center get higher amplitudes
            center_factor = 1.0 - 2.0 * abs(pos) / 0.5
            base_height = np.random.uniform(1.0, 2.0) * max(0.5, center_factor)

            # Widths inversely related to height but with some randomness
            # To promote beneficial convolution behavior
            base_width = 0.01 + 0.02 * np.exp(-abs(pos) * 2.0)  # Narrower near center
            width = base_width * np.random.uniform(0.8, 1.2)

            peak_heights.append(base_height)
            peak_widths.append(width)

        # Create Gaussian peaks with optimized characteristics
        for center, height, width in zip(peak_positions, peak_heights, peak_widths):
            gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
            f_values += gaussian

        # Apply additional structure to encourage flat autoconvolution profiles
        # Add some polynomial terms to create more varied convolution behavior
        poly_term = 0.1 * (x**2) + 0.05 * (x**4)
        f_values += np.maximum(poly_term, 0) * np.random.uniform(0.5, 1.5)

        # Normalize and apply final adjustments
        if np.max(f_values) > 0:
            f_values = f_values / np.max(f_values) * 1.5

        # Add more controlled random variation to break symmetry without over扰
        noise = np.random.normal(0, 0.005, n_steps)  # Reduced noise
        f_values = np.maximum(f_values + noise, 0)

        # Apply gentle smoothing to reduce sharp discontinuities that could cause numerical issues
        if n_steps > 50:
            f_values = gaussian_filter1d(f_values, sigma=0.3, mode='nearest')

        return f_values.tolist()

    def improve_with_local_search(self, initial_func: List[float], max_evals: int = 500) -> List[float]:
        """Apply local refinement to improve initial solution"""
        try:
            # Convert to numpy for easier manipulation
            current_func = np.array(initial_func)

            # Simple gradient-like improvement
            improved_func = current_func.copy()

            # Apply small adjustments to reduce peak dominance
            # This helps balance ||g||₂² against ||g||₁·||g||∞
            for i in range(len(improved_func)):
                # Slightly reduce high values to prevent excessive peaks in autoconvolution
                if improved_func[i] > 0.5:
                    improved_func[i] *= 0.95

            # Additional smoothing to reduce numerical artifacts
            smoothed = gaussian_filter1d(improved_func, sigma=0.5)
            smoothed = np.maximum(smoothed, 0)

            return smoothed.tolist()

        except Exception:
            return initial_func

    def create_balanced_distribution(self, n_steps: int) -> List[float]:
        """Create a balanced distribution that typically performs well"""
        # Use a combination of Gaussian and polynomial shapes
        x = np.linspace(-0.25, 0.25, n_steps)

        # Create a base shape with multiple peaks
        base_shape = (0.5 * np.exp(-x**2 / 0.02) +
                     0.3 * np.exp(-((x - 0.1)**2) / 0.01) +
                     0.2 * np.exp(-((x + 0.1)**2) / 0.01))

        # Normalize and add some randomness
        base_shape = base_shape / np.max(base_shape) * 0.8

        # Add small random variations
        noise = np.random.normal(0, 0.02, n_steps)
        final_shape = np.maximum(base_shape + noise, 0)

        return final_shape.tolist()

    def construct_function(self) -> List[float]:
        """Main function to construct step-function with high C2 value."""
        start_time = time.time()

        # Determine number of steps with time budget consideration
        n_steps = min(1000, max(100, 500 + int(np.random.randint(0, 200) * 2)))

        # Phase 1: Create baseline function with good structure
        if time.time() - start_time > 80:  # Leave margin for processing
            return [0.5] * n_steps

        # Start with a balanced distribution
        best_function = self.create_balanced_distribution(n_steps)

        # Phase 2: Local refinement
        if time.time() - start_time > 85:
            return best_function

        refined_function = self.improve_with_local_search(best_function)

        # Phase 3: Final validation and adjustment
        c2_score = self.compute_c2(refined_function)

        # If we're close to timeout, just return what we have
        if time.time() - start_time > 88:
            return refined_function

        # Apply one final adjustment if needed
        if c2_score < 0.1:  # Very low score - try alternative
            alternative = self.create_initial_function(n_steps)
            alt_c2 = self.compute_c2(alternative)
            if alt_c2 > c2_score:
                refined_function = alternative

        return refined_function

def construct_function() -> List[float]:
    """Wrapper function that maintains interface compatibility"""
    optimizer = AutoCorrelationOptimizer()
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")