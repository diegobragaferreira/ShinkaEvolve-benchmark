# EVOLVE-BLOCK-START

import numpy as np
from typing import List, Tuple, Optional
import time

class AutoconvolutionComputations:
    """Handles all autoconvolution-related computations"""
    
    @staticmethod
    def compute_autoconvolution(f_values: List[float]) -> np.ndarray:
        """Compute the autoconvolution g = f * f of step function f."""
        n = len(f_values)
        if n == 0:
            return np.array([])

        f_array = np.array(f_values)
        g = np.convolve(f_array, f_array, mode='full')
        g = g[n-1:-(n-1)] if n > 1 else g
        return g

class NormComputations:
    """Handles all norm computations for C2 calculation"""
    
    @staticmethod
    def compute_norms(g_values: np.ndarray) -> Tuple[float, float, float]:
        """Compute the three required norms for C2 calculation."""
        if len(g_values) == 0:
            return 0.0, 0.0, 0.0

        # ||g||₂² using trapezoidal-like piecewise linear integration
        if len(g_values) <= 1:
            norm_2_sq = g_values[0]**2 if len(g_values) > 0 else 0.0
        else:
            norm_2_sq = 0.0
            for i in range(len(g_values)-1):
                h = 1.0  # Assuming unit spacing
                norm_2_sq += (h/3.0) * (g_values[i]**2 + g_values[i]*g_values[i+1] + g_values[i+1]**2)

        # ||g||₁: L1 norm, approximated as sum(|g|) / (len(g) + 1)
        if len(g_values) > 0:
            norm_1 = np.sum(np.abs(g_values)) / (len(g_values) + 1)
        else:
            norm_1 = 0.0

        # ||g||∞: Infinity-norm
        norm_inf = np.max(np.abs(g_values)) if len(g_values) > 0 else 0.0

        return norm_2_sq, norm_1, norm_inf

class StepFunctionOptimizer:
    """Main optimizer class that orchestrates the step function construction"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.autoconvolver = AutoconvolutionComputations()
        self.norm_computer = NormComputations()
        
        # Configuration parameters
        self.domain_width = 0.5
        self.domain_center = 0.0
        self.default_step_count = 1000
    
    def compute_c2(self, f_values: List[float]) -> float:
        """Compute C2 for given step function values."""
        g = self.autoconvolver.compute_autoconvolution(f_values)
        norm_2_sq, norm_1, norm_inf = self.norm_computer.compute_norms(g)

        # Avoid division by zero
        if norm_1 == 0 or norm_inf == 0:
            return 0.0

        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2

    def gaussian_peak_function(self, x: np.ndarray, peak_params: List[float]) -> np.ndarray:
        """Generate a function composed of multiple Gaussian peaks."""
        result = np.zeros_like(x)
        for i in range(0, len(peak_params), 3):
            amp, center, width = peak_params[i], peak_params[i+1], peak_params[i+2]
            width = max(width, 1e-6)  # Ensure positive width
            result += amp * np.exp(-0.5 * ((x - center) / width)**2)
        return result

    def enforce_peak_spacing(self, peak_params: List[float], 
                           min_distance_ratio: float = 0.1) -> None:
        """Enforce minimum distance between Gaussian peaks to prevent narrow autoconvolution."""
        if len(peak_params) < 3:
            return

        # Group peaks by their parameters [amp, center, width]
        peaks = []
        for i in range(0, len(peak_params), 3):
            peaks.append([peak_params[i], peak_params[i+1], peak_params[i+2]])

        # Sort by center position
        peaks.sort(key=lambda x: x[1])

        # Ensure minimum spacing
        min_distance = min_distance_ratio * self.domain_width
        for i in range(1, len(peaks)):
            prev_center = peaks[i-1][1]
            curr_center = peaks[i][1]
            distance = abs(curr_center - prev_center)

            if distance < min_distance:
                # Adjust position of current peak
                offset = min_distance - distance
                if curr_center > prev_center:
                    peaks[i][1] += offset
                else:
                    peaks[i][1] -= offset

        # Put them back into flat list
        for i, (amp, center, width) in enumerate(peaks):
            peak_params[i*3] = amp
            peak_params[i*3 + 1] = center
            peak_params[i*3 + 2] = width

    def adaptive_gaussian_construction(self, n_steps: int = 1000) -> List[float]:
        """
        Advanced Gaussian peak construction with adaptive parameters.
        """
        # Domain setup
        domain_width = 0.5
        step_width = domain_width / n_steps

        # Start with a configurable number of peaks
        n_peaks = 8

        # Initialize peaks with logarithmic spacing to avoid clustering at edges
        peak_positions = []
        for i in range(n_peaks):
            # Logarithmic distribution to concentrate points near center
            ratio = (i + 1) / (n_peaks + 1)
            # Apply exponential mapping with power to concentrate more points near center
            pos = self.domain_center + (ratio ** 1.8) * domain_width/2
            if i % 2 == 0:
                pos = -pos  # Alternate sides for symmetry
            peak_positions.append(pos)

        # Initialize peak parameters [amplitude, center, width]
        peak_params = []
        for i in range(n_peaks):
            amplitude = 50.0 + np.random.random() * 50.0
            center = peak_positions[i]
            width = 0.03 + np.random.random() * 0.04
            peak_params.extend([amplitude, center, width])

        # Build the function progressively
        domain_points = np.linspace(-domain_width/2, domain_width/2, n_steps)

        # Adaptive optimization loop with better convergence control
        best_c2 = -1.0
        best_params = list(peak_params)
        best_function = None

        max_iterations = 300
        for iteration in range(max_iterations):
            # Create function from current peak parameters
            func_values = self.gaussian_peak_function(domain_points, peak_params)

            # Convert to step function by taking samples
            step_values = func_values.tolist()

            # Compute C2
            c2_val = self.compute_c2(step_values)

            # Check if this is our best result so far
            if c2_val > best_c2:
                best_c2 = c2_val
                best_params = list(peak_params)
                best_function = step_values.copy()

            # Stop if we're getting close to convergence
            if iteration > 10 and abs(c2_val - best_c2) < 1e-8:
                break

            # Apply adaptive adjustments to peak parameters
            for i in range(0, len(peak_params), 3):
                # Perturb amplitude with moderate change
                if i < len(peak_params):
                    old_amp = peak_params[i]
                    change_factor = 1.0 + np.random.normal(0, 0.1)
                    new_amp = max(0, old_amp * change_factor)
                    peak_params[i] = new_amp

                # Perturb width slightly
                if i+2 < len(peak_params):
                    old_width = peak_params[i+2]
                    change_factor = 1.0 + np.random.normal(0, 0.1)
                    new_width = max(0.001, old_width * change_factor)
                    peak_params[i+2] = new_width

            # Enforce minimum spacing between peaks
            self.enforce_peak_spacing(peak_params, 0.1)

            # Occasionally do larger changes to explore more space
            if iteration > 50 and iteration % 25 == 0:
                for i in range(0, len(peak_params), 3):
                    if i < len(peak_params):
                        change_factor = 1.0 + np.random.normal(0, 0.2)
                        new_amp = max(0, peak_params[i] * change_factor)
                        peak_params[i] = new_amp

            # Reduce amplitudes if autoconvolution becomes too peaked
            if iteration > 30 and np.random.rand() < 0.1:
                func_values = self.gaussian_peak_function(domain_points, peak_params)
                g = self.autoconvolver.compute_autoconvolution(func_values.tolist())
                if len(g) > 0:
                    norm_inf = np.max(np.abs(g))
                    if norm_inf > 150:
                        for j in range(0, len(peak_params), 3):
                            if j < len(peak_params):
                                peak_params[j] *= 0.97

        # Final refinement - use the best parameters found
        final_func_values = self.gaussian_peak_function(domain_points, best_params)
        return final_func_values.tolist()

    def adaptive_refinement(self, f_values: List[float], max_iterations: int = 300,
                           initial_step_size: float = 0.1, min_improvement: float = 1e-6) -> List[float]:
        """
        Improved adaptive refinement with better convergence criteria and step size control.
        """
        current_f = list(f_values)
        current_c2 = self.compute_c2(current_f)

        prev_c2 = current_c2
        improvement_count = 0
        step_size = initial_step_size

        iteration = 0
        while iteration < max_iterations:
            modified_f = list(current_f)
            idx = np.random.randint(len(modified_f))

            delta = np.random.normal(0, step_size * 0.5)
            modified_f[idx] = max(0.0, modified_f[idx] + delta)

            test_f = list(modified_f)
            test_c2 = self.compute_c2(test_f)

            if test_c2 > current_c2:
                current_f = test_f
                current_c2 = test_c2
                improvement_count = 0

                if test_c2 - prev_c2 > min_improvement * 10:
                    step_size = min(initial_step_size, step_size * 1.1)
            else:
                improvement_count += 1

            if improvement_count > 3:
                step_size *= 0.95
                improvement_count = 0
            elif improvement_count == 0 and abs(test_c2 - current_c2) > min_improvement:
                step_size = min(initial_step_size, step_size * 1.05)

            prev_c2 = current_c2
            iteration += 1

            if improvement_count > 15:
                break

        return current_f

    def construct_function(self) -> List[float]:
        """
        Main function to construct step-function with high C2 value.
        Uses advanced adaptive Gaussian construction with multiple strategies.
        """
        best_c2 = 0.0
        best_function = []

        # Strategy 1: Multi-start with different peak configurations
        strategies = [
            {"n_peaks": 5, "n_steps": 800},
            {"n_peaks": 8, "n_steps": 1000},
            {"n_peaks": 10, "n_steps": 1200},
        ]

        for strategy in strategies:
            try:
                func = self.adaptive_gaussian_construction(strategy["n_steps"])
                c2_val = self.compute_c2(func)

                if c2_val > best_c2:
                    best_c2 = c2_val
                    best_function = func
            except Exception:
                continue

        # Strategy 2: If first strategy failed, try refined approach
        if len(best_function) == 0:
            try:
                func = self.adaptive_gaussian_construction(1500)
                c2_val = self.compute_c2(func)

                if c2_val > best_c2:
                    best_c2 = c2_val
                    best_function = func
            except Exception:
                pass

        # Strategy 3: Final fallback to simple approach
        if len(best_function) == 0:
            n_steps = 500
            base_f = [10.0] * n_steps
            best_function = base_f

        return best_function

def construct_function() -> List[float]:
    """Main entry point function for constructing step function with high C2 value."""
    optimizer = StepFunctionOptimizer(seed=42)
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")