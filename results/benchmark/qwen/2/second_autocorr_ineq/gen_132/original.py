# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List, Tuple, Optional
import warnings

class PeakGenerator:
    """Generates strategic peak distributions for optimal autoconvolution properties."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_multi_scale_peaks(self, n_steps: int, peak_count: Optional[int] = None) -> List[Tuple[float, float, float]]:
        """
        Generate multi-scale peaks with strategic placement and characteristics.

        Args:
            n_steps: Number of steps in the function
            peak_count: Optional number of peaks to generate

        Returns:
            List of (position, height, width) tuples
        """
        if peak_count is None:
            peak_count = max(10, min(50, n_steps // 100))

        x = np.linspace(-0.25, 0.25, n_steps)
        peaks = []

        # Scale 1: Fine scale peaks
        fine_count = max(3, min(15, peak_count // 3))
        fine_positions = np.random.uniform(-0.05, 0.05, fine_count)
        fine_heights = np.random.uniform(1.5, 2.5, fine_count)
        fine_widths = np.random.uniform(0.005, 0.015, fine_count)

        # Scale 2: Medium scale peaks
        medium_count = max(5, min(25, peak_count // 2))
        medium_positions = np.random.uniform(-0.15, 0.15, medium_count)
        medium_heights = np.random.uniform(1.2, 2.0, medium_count)
        medium_widths = np.random.uniform(0.015, 0.035, medium_count)

        # Scale 3: Coarse scale peaks
        coarse_count = max(2, min(15, peak_count // 6))
        coarse_positions = np.random.choice([-0.2, -0.18, -0.16, -0.14, -0.12, -0.1,
                                           -0.08, -0.06, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2],
                                         coarse_count)
        coarse_heights = np.random.uniform(1.0, 1.8, coarse_count)
        coarse_widths = np.random.uniform(0.025, 0.055, coarse_count)

        # Combine and filter peaks
        all_positions = np.concatenate([fine_positions, medium_positions, coarse_positions])
        all_heights = np.concatenate([fine_heights, medium_heights, coarse_heights])
        all_widths = np.concatenate([fine_widths, medium_widths, coarse_widths])

        # Filter for minimum separation
        filtered_peaks = self._filter_peaks(all_positions, all_heights, all_widths)

        # Enhance peak qualities
        enhanced_peaks = self._enhance_peaks(filtered_peaks)

        return enhanced_peaks

    def _filter_peaks(self, positions: np.ndarray, heights: np.ndarray, widths: np.ndarray) -> List[Tuple[float, float, float]]:
        """Filter peaks by minimum spatial separation."""
        filtered_peaks = []
        min_gap = 0.01

        for pos, height, width in zip(positions, heights, widths):
            valid = True
            for existing_pos, _, _ in filtered_peaks:
                if abs(pos - existing_pos) < min_gap:
                    valid = False
                    break
            if valid:
                filtered_peaks.append((pos, height, width))

        return filtered_peaks

    def _enhance_peaks(self, peaks: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Enhance peak characteristics based on position."""
        enhanced = []
        for pos, height, width in peaks:
            if abs(pos) > 0.15:
                height *= 0.8
            enhanced.append((pos, height, width))
        return enhanced

class FunctionBuilder:
    """Constructs functions from peak specifications."""

    def __init__(self):
        pass

    def build_from_peaks(self, peaks: List[Tuple[float, float, float]], n_steps: int) -> np.ndarray:
        """Build function from peak specifications."""
        x = np.linspace(-0.25, 0.25, n_steps)
        f_values = np.zeros(n_steps)

        # Apply all peaks
        for pos, height, width in peaks:
            gaussian = height * np.exp(-0.5 * ((x - pos) / width)**2)
            f_values += gaussian

        # Add supplementary structure
        self._add_supplementary_structure(f_values, x)

        # Ensure non-negativity and normalize
        f_values = np.maximum(f_values, 0)
        if np.max(f_values) > 0:
            f_values = f_values / np.max(f_values) * 1.8

        return f_values

    def _add_supplementary_structure(self, f_values: np.ndarray, x: np.ndarray):
        """Add supplementary structure for better autoconvolution properties."""
        n_steps = len(f_values)
        for i in range(0, n_steps, max(1, n_steps // 40)):
            if np.random.random() > 0.8:
                bump_center = x[i]
                bump_height = np.random.uniform(0.05, 0.3)
                bump_width = np.random.uniform(0.005, 0.015)
                bump = bump_height * np.exp(-0.5 * ((x - bump_center) / bump_width)**2)
                f_values += bump

class Optimizer:
    """Performs local optimization on functions."""

    def __init__(self):
        pass

    def optimize(self, func_vals: List[float]) -> List[float]:
        """Perform local optimization using hill-climbing."""
        current_func = np.array(func_vals)
        best_c2 = self._compute_c2(current_func)
        best_func = current_func.copy()

        # Multi-pass optimization
        for _ in range(3):
            # Hill climbing pass
            for _ in range(100):
                test_func = current_func.copy()
                idx = np.random.randint(0, len(test_func))
                adjustment = np.random.normal(0, 0.03)
                test_func[idx] = max(0, test_func[idx] + adjustment)
                test_c2 = self._compute_c2(test_func)
                if test_c2 > best_c2:
                    best_c2 = test_c2
                    best_func = test_func.copy()

            # Second pass with smaller adjustments
            for _ in range(50):
                test_func = best_func.copy()
                idx = np.random.randint(0, len(test_func))
                adjustment = np.random.normal(0, 0.02)
                test_func[idx] = max(0, test_func[idx] + adjustment)
                test_c2 = self._compute_c2(test_func)
                if test_c2 > best_c2:
                    best_c2 = test_c2
                    best_func = test_func.copy()

            current_func = best_func.copy()

        return best_func.tolist()

    def _compute_c2(self, func_vals: np.ndarray) -> float:
        """Compute C₂ value with numerical stability."""
        f = np.array(func_vals)

        # Autoconvolution using convolution
        g = np.convolve(f, f, mode='full')
        g = g[len(g)//2:]

        # Adjust for correct length
        if len(g) > len(f):
            g = g[:len(f)]

        # Compute norms
        dx = 0.5 / (len(f) - 1) if len(f) > 1 else 0.5
        norm_2_sq = 0
        for i in range(len(g)-1):
            area = (g[i]**2 + g[i+1]**2) * dx / 2
            norm_2_sq += area

        norm_1 = np.sum(np.abs(g)) * dx
        norm_inf = np.max(np.abs(g))

        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0

        return norm_2_sq / (norm_1 * norm_inf)

class HarmonicPeakOptimizer:
    """Main optimizer class that orchestrates the entire process."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.peak_generator = PeakGenerator(seed)
        self.function_builder = FunctionBuilder()
        self.optimizer = Optimizer()

    def construct_function(self, n_steps: Optional[int] = None) -> List[float]:
        """
        Main function to construct step function with high C2 value.

        Args:
            n_steps: Optional number of steps to use

        Returns:
            List of step heights
        """
        if n_steps is None:
            n_steps = np.random.randint(2000, 8000)

        # Generate peaks
        peaks = self.peak_generator.generate_multi_scale_peaks(n_steps)

        # Build function
        f_values = self.function_builder.build_from_peaks(peaks, n_steps)

        # Apply smoothing
        f_values = self._smooth_function(f_values, n_steps)

        # Convert to list
        f_list = f_values.tolist()

        # Optimize
        try:
            optimized_func = self.optimizer.optimize(f_list)
            final_func = np.array(optimized_func)

            # Add final noise
            noise = np.random.normal(0, 0.005, len(final_func))
            final_func = final_func + noise
            final_func = np.maximum(final_func, 0)

            return final_func.tolist()

        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")
            return f_list

    def _smooth_function(self, f_values: np.ndarray, n_steps: int) -> np.ndarray:
        """Apply smoothing to reduce sharp transitions."""
        window_size = min(51, max(3, n_steps // 150))
        if window_size % 2 == 0:
            window_size += 1
        if window_size > 1:
            window = np.ones(window_size) / window_size
            f_values = np.convolve(f_values, window, mode='same')
        return f_values

def construct_function() -> List[float]:
    """
    Harmonic peak optimizer for maximizing C₂ constant.
    Uses frequency-domain analysis and constrained optimization.
    """
    optimizer = HarmonicPeakOptimizer(seed=42)
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")