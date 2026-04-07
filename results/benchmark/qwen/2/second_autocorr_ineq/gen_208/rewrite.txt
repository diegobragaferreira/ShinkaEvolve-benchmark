# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from numba import jit
import time
import warnings
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class NormResults:
    """Container for autoconvolution norm results."""
    norm_2_sq: float
    norm_1: float
    norm_inf: float

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    function_values: List[float]
    c2_score: float

class AutoconvolutionComputer:
    """Handles all autoconvolution norm computations with numerical stability."""
    
    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution_norms(f_values: List[float]) -> NormResults:
        """
        Fast computation of the three norms needed for C2 calculation using piecewise linear integration.
        """
        if not f_values:
            return NormResults(0.0, 1e-15, 1e-15)
            
        # Convert to numpy array for easier manipulation
        f = np.array(f_values, dtype=np.float64)
        n_steps = len(f)

        if n_steps == 0:
            return NormResults(0.0, 1e-15, 1e-15)

        # Step width
        dx = 0.5 / n_steps

        # Compute autoconvolution using discrete convolution
        g = np.convolve(f, f, mode='full')
        # Trim g to the correct size (this accounts for the convolution)
        g = g[len(f)-1:2*len(f)-1]

        # Compute L2 norm squared using piecewise linear integration
        norm_2_squared = 0.0
        for i in range(len(g)-1):
            # Trapezoidal-like integration for quadratic function
            # Using formula for integral of ax^2 + bx + c over [x0,x1]
            # But here we approximate with piecewise linear segments
            # So we use: (dx/3)(y0^2 + y0*y1 + y1^2)
            y0, y1 = g[i], g[i+1]
            norm_2_squared += (dx/3) * (y0**2 + y0*y1 + y1**2)

        # L1 norm (sum of absolute values)
        norm_1 = np.sum(np.abs(g))

        # Infinity norm
        norm_inf = np.max(np.abs(g))

        # Handle numerical edge cases
        if norm_1 <= 1e-15:
            norm_1 = 1e-15
        if norm_inf <= 1e-15:
            norm_inf = 1e-15

        return NormResults(norm_2_squared, norm_1, norm_inf)

class StepFunctionGenerator:
    """Handles construction of optimized step functions with various strategies."""
    
    def __init__(self):
        self._seed = 42
        np.random.seed(self._seed)
    
    def _generate_peak_positions(self, n_peaks: int, domain_width: float = 0.5) -> np.ndarray:
        """Generate peak positions using multi-scale logarithmic distribution."""
        # Create multiple scales of peaks
        scales = np.logspace(np.log10(0.01), np.log10(0.25), 5)  # 5 different scales
        
        all_positions = []
        
        for scale in scales:
            # Determine how many peaks per scale
            n_per_scale = max(2, int(n_peaks * scale / 0.25))
            # Generate positions with logarithmic spacing within this scale
            positions = np.logspace(np.log10(scale * 0.1), np.log10(scale), n_per_scale)
            # Mirror to create symmetric distribution
            positions = np.concatenate([-positions[::-1], positions])
            # Filter to domain
            positions = positions[(positions >= -0.25) & (positions <= 0.25)]
            all_positions.extend(positions)
        
        # Remove duplicates and sort
        all_positions = np.unique(all_positions)
        
        # Ensure we don't exceed our target number of peaks
        if len(all_positions) > n_peaks:
            # Select a subset using weighted sampling toward center
            weights = np.exp(-10 * all_positions**2)  # Higher weight for center positions
            weights = weights / np.sum(weights)
            selected_indices = np.random.choice(len(all_positions), size=n_peaks, p=weights, replace=False)
            all_positions = all_positions[selected_indices]
            
        return all_positions
    
    def _enforce_minimum_spacing(self, positions: np.ndarray, domain_width: float) -> np.ndarray:
        """Enforce minimum spacing between peaks."""
        if len(positions) <= 1:
            return positions
            
        sorted_positions = np.sort(positions)
        filtered_positions = [sorted_positions[0]]
        
        # Calculate adaptive minimum distance based on peak density
        distances = np.diff(sorted_positions)
        if len(distances) > 0:
            avg_distance = np.mean(distances)
            min_spacing = max(0.01 * domain_width, avg_distance * 0.7)
        else:
            min_spacing = 0.02 * domain_width
            
        # Apply spacing enforcement
        for i in range(1, len(sorted_positions)):
            if sorted_positions[i] - filtered_positions[-1] >= min_spacing:
                filtered_positions.append(sorted_positions[i])
                
        return np.array(filtered_positions)
    
    def _calculate_peak_widths(self, positions: np.ndarray) -> np.ndarray:
        """Calculate adaptive widths for peaks based on their positions and density."""
        if len(positions) == 0:
            return np.array([])
            
        # Calculate peak densities
        if len(positions) > 1:
            distances = np.diff(np.abs(positions))
            if len(distances) > 0:
                avg_distance = np.mean(distances)
                # Higher density = narrower widths, lower density = wider widths
                peak_densities = np.clip(1.0 / (distances + 1e-8), 0.5, 2.0)
                # Extend to full length
                peak_densities = np.append(peak_densities, peak_densities[-1] if len(peak_densities) > 0 else 1.0)
            else:
                peak_densities = np.ones(len(positions))
        else:
            peak_densities = np.ones(len(positions))
            
        # Calculate adaptive widths
        base_widths = 0.02 + 0.06 * (1.0 - np.abs(positions) / 0.25)
        widths = base_widths / (peak_densities * 0.5 + 0.5)
        widths = np.clip(widths, 0.01, 0.15)  # Keep within reasonable bounds
        
        return widths
    
    def _create_gaussian_peaks(self, positions: np.ndarray, amplitudes: np.ndarray, 
                              widths: np.ndarray, x_domain: np.ndarray) -> np.ndarray:
        """Create Gaussian peaks with specified parameters."""
        f = np.zeros_like(x_domain)
        for i in range(len(positions)):
            amp = amplitudes[i]
            pos = positions[i]
            width = widths[i]
            if width > 1e-6:
                f += amp * np.exp(-0.5 * ((x_domain - pos) / width)**2)
        return f
    
    def _apply_smoothing(self, function_values: np.ndarray, n_steps: int) -> np.ndarray:
        """Apply smoothing to reduce numerical artifacts."""
        if n_steps <= 100:
            return function_values
            
        try:
            from scipy.signal import gaussian, convolve
            # Create a Gaussian kernel for smoothing
            kernel_size = min(21, n_steps // 20 + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = gaussian(kernel_size, std=kernel_size/6.0)
            kernel = kernel / np.sum(kernel)  # Normalize
            # Apply convolution to smooth without losing peak information
            return convolve(function_values, kernel, mode='same')
        except Exception:
            # Fallback to simple moving average if scipy unavailable
            window_size = min(15, n_steps // 10)
            if window_size % 2 == 0:
                window_size += 1
            if window_size > 1:
                return np.convolve(function_values, np.ones(window_size)/window_size, mode='same')
            return function_values
    
    def generate_adaptive_function(self, n_steps: int) -> np.ndarray:
        """
        Generate optimized step function with adaptive Gaussian peak placement.
        """
        # Set seed for reproducibility
        np.random.seed(self._seed)
        
        # Multi-scale approach: create peaks at different scales
        domain_width = 0.5
        n_peaks = max(20, min(200, n_steps // 25))
        
        # Generate peak positions
        all_positions = self._generate_peak_positions(n_peaks, domain_width)
        
        # Enforce minimum spacing
        all_positions = self._enforce_minimum_spacing(all_positions, domain_width)
        
        # Generate amplitudes with structured noise
        amplitudes = np.exp(-15 * all_positions**2)  # Stronger center concentration
        
        # Add structured noise and modulation
        noise = np.random.normal(0, 0.2, len(all_positions))
        modulation = 0.1 * np.sin(10 * np.pi * all_positions) * np.exp(-all_positions**2/0.05)
        amplitudes += noise + modulation
        
        # Ensure non-negative amplitudes
        amplitudes = np.maximum(amplitudes, 0)
        
        # Test autoconvolution quality and adjust if needed
        test_amplitudes = amplitudes.copy()
        if np.sum(test_amplitudes) > 0:
            test_amplitudes = test_amplitudes / np.sum(test_amplitudes) * 10
            
            # Quick test of autoconvolution
            test_g = np.convolve(test_amplitudes, test_amplitudes, mode='full')
            test_g = test_g[len(test_amplitudes)-1:2*len(test_amplitudes)-1]
            test_norm_1 = np.sum(np.abs(test_g))
            test_norm_inf = np.max(np.abs(test_g)) if np.max(np.abs(test_g)) > 0 else 1e-15
            
            # If the autoconvolution shows signs of being too spiked, reduce amplitudes
            if test_norm_inf > 5 * test_norm_1 and test_norm_1 > 0:
                # Scale down amplitudes to reduce peakiness
                amplitudes = amplitudes * 0.7
        
        # Final normalization
        if np.sum(amplitudes) > 0:
            amplitudes = amplitudes / np.sum(amplitudes) * 10
        
        # Convert to step function by interpolating to desired resolution
        x_domain = np.linspace(-0.25, 0.25, n_steps)
        
        # Calculate adaptive widths
        widths = self._calculate_peak_widths(all_positions)
        
        # Create function with adaptive widths
        step_function = self._create_gaussian_peaks(all_positions, amplitudes, widths, x_domain)
        
        # Apply smoothing to reduce numerical artifacts while preserving structure
        step_function = self._apply_smoothing(step_function, n_steps)
        
        # Add fine-grained modulation to break symmetries
        fine_modulation = 0.03 * np.sin(40 * np.pi * x_domain) * np.exp(-x_domain**2/0.04)
        fine_modulation += 0.02 * np.cos(25 * np.pi * x_domain) * np.exp(-x_domain**2/0.06)
        step_function += fine_modulation
        
        # Ensure non-negativity and normalize
        step_function = np.maximum(step_function, 0)
        if np.sum(step_function) > 0:
            # Use a more conservative normalization to avoid over-amplification
            step_function = step_function / np.sum(step_function) * 8
        
        return step_function

class MultiStrategyOptimizer:
    """Manages multiple optimization strategies and selects the best result."""
    
    def __init__(self):
        self.computer = AutoconvolutionComputer()
        self.generator = StepFunctionGenerator()
        self._seed = 42
        np.random.seed(self._seed)
        
    def evaluate_function(self, function_values: List[float]) -> Optional[float]:
        """Evaluate a function and return its C2 score."""
        try:
            norms = self.computer.compute_autoconvolution_norms(function_values)
            if norms.norm_1 <= 1e-15 or norms.norm_inf <= 1e-15:
                return None
            return norms.norm_2_sq / (norms.norm_1 * norms.norm_inf)
        except Exception as e:
            warnings.warn(f"Evaluation error: {str(e)}")
            return None
    
    def optimize_with_strategy(self, n_steps: int, max_attempts: int = 10) -> OptimizationResult:
        """
        Optimize function using multiple strategies.
        """
        best_c2 = -1.0
        best_function = None
        
        for attempt in range(max_attempts):
            try:
                # Generate function using adaptive Gaussian approach
                function_values = self.generator.generate_adaptive_function(n_steps)
                
                # Evaluate function
                c2 = self.evaluate_function(function_values)
                
                if c2 is not None and c2 > best_c2:
                    best_c2 = c2
                    best_function = function_values.tolist()
                    
            except Exception as e:
                warnings.warn(f"Strategy attempt failed: {str(e)}")
                continue
                
        # Fallback if no valid results
        if best_function is None:
            n_steps = 1000
            best_function = [1.0] * n_steps
            best_c2 = 0.0
            
        return OptimizationResult(best_function, best_c2)

def construct_function() -> List[float]:
    """
    Main function to construct optimized step function with high C2 value.
    This function orchestrates the optimization process.
    """
    # Initialize optimizer
    optimizer = MultiStrategyOptimizer()
    
    # Multi-attempt selection to maximize C2
    best_c2 = -1.0
    best_function = None
    start_time = time.time()
    
    # Set maximum attempts to balance quality vs. time constraints
    max_attempts = 50
    
    for attempt in range(max_attempts):
        # Early termination if time is running out
        if time.time() - start_time > 85:  # Leave 5 seconds for cleanup
            break
            
        # Try different number of steps to find optimal
        n_steps = np.random.randint(1000, 5000)
        
        # Optimize with strategy
        try:
            result = optimizer.optimize_with_strategy(n_steps, max_attempts=5)
            
            # Keep the best function
            if result.c2_score > best_c2:
                best_c2 = result.c2_score
                best_function = result.function_values
                
        except Exception as e:
            warnings.warn(f"Optimization attempt failed: {str(e)}")
            continue
    
    # Return the best function found, or fallback
    if best_function is not None:
        return best_function
    else:
        # Fallback to a simpler construction
        n_steps = 1000
        f_values = [1.0] * n_steps
        return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")