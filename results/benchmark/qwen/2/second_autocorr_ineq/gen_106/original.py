# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List, Tuple, Any
import time

class AutoCorrelationOptimizer:
    """Modular optimizer for constructing step functions to maximize C2 constant."""
    
    def __init__(self, seed: int = 42, max_time: float = 90.0):
        """Initialize optimizer with deterministic behavior."""
        self.seed = seed
        self.max_time = max_time
        self.start_time = time.time()
        
        np.random.seed(seed)
        random.seed(seed)
        
    def _time_remaining(self) -> float:
        """Check if there's time left for computation."""
        return self.max_time - (time.time() - self.start_time)
        
    def _compute_autoconvolution(self, f: np.ndarray) -> np.ndarray:
        """Compute autoconvolution g = f * f efficiently."""
        # Use numpy's convolve for speed
        g = np.convolve(f, f, mode='full')
        g = g[len(g)//2:]  # Keep only positive lags
        
        # Truncate if necessary to match input length
        if len(g) > len(f):
            g = g[:len(f)]
            
        return g
        
    def _compute_norms(self, g: np.ndarray) -> Tuple[float, float, float]:
        """Compute the three required norms for C2 calculation."""
        # L1 norm (absolute integral)
        norm_1 = np.sum(np.abs(g)) / (len(g) + 1)
        
        # L2 squared norm (squared integral)
        norm_2_sq = np.sum(g**2) * (0.5 / len(g))
        
        # L-infinity norm (maximum absolute value)
        norm_inf = np.max(np.abs(g))
        
        return norm_1, norm_2_sq, norm_inf
        
    def _compute_c2(self, f: np.ndarray) -> float:
        """Compute C2 value for given function."""
        try:
            # Compute autoconvolution
            g = self._compute_autoconvolution(f)
            
            # Compute norms
            norm_1, norm_2_sq, norm_inf = self._compute_norms(g)
            
            # Avoid division by zero
            if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                return 0.0
                
            # Compute C2
            c2 = norm_2_sq / (norm_1 * norm_inf)
            return c2
            
        except Exception:
            return 0.0
    
    def _generate_base_function(self, n_steps: int) -> np.ndarray:
        """Generate base multi-peak Gaussian structure."""
        x = np.linspace(-0.25, 0.25, n_steps)
        base_function = np.zeros_like(x)
        
        # Use logarithmic spacing for peak positions
        num_peaks = np.random.randint(10, 30)
        log_positions = np.logspace(np.log10(0.01), np.log10(0.24), num_peaks)
        peak_positions = np.concatenate([log_positions, -log_positions[::-1]])
        peak_positions = peak_positions[(peak_positions <= 0.25) & (peak_positions >= -0.25)]
        
        # Ensure minimum gap between peaks
        min_gap = 0.1 * 0.5
        safe_positions = []
        for pos in sorted(peak_positions):
            if not safe_positions or abs(pos - safe_positions[-1]) >= min_gap:
                safe_positions.append(pos)
                
        num_peaks = len(safe_positions)
        
        # Construct peaks with optimized parameters
        for i in range(num_peaks):
            peak_center = safe_positions[i]
            
            # Adjust peak height based on position
            if abs(peak_center) > 0.15:
                peak_height = np.random.uniform(1.0, 1.5)
            else:
                peak_height = np.random.uniform(1.2, 2.0)
                
            # Use narrower widths for central peaks
            if abs(peak_center) < 0.05:
                peak_width = np.random.uniform(0.015, 0.03)
            elif abs(peak_center) < 0.15:
                peak_width = np.random.uniform(0.025, 0.05)
            else:
                peak_width = np.random.uniform(0.03, 0.07)
                
            # Create Gaussian peak
            gaussian_peak = peak_height * np.exp(-0.5 * ((x - peak_center) / peak_width)**2)
            base_function += gaussian_peak
            
        # Add additional structure with controlled randomness
        for i in range(0, len(x), max(1, len(x)//20)):
            if np.random.random() > 0.8:
                bump_center = x[i]
                bump_height = np.random.uniform(0.05, 0.3)
                bump_width = np.random.uniform(0.005, 0.015)
                bump = bump_height * np.exp(-0.5 * ((x - bump_center) / bump_width)**2)
                base_function += bump
                
        return base_function
        
    def _enhance_function(self, base_function: np.ndarray, n_steps: int) -> np.ndarray:
        """Apply enhancements like smoothing and noise to improve function quality."""
        # Ensure non-negative values
        enhanced_function = np.maximum(base_function, 0)
        
        # Normalize to avoid extreme values
        if np.max(enhanced_function) > 0:
            enhanced_function = enhanced_function / np.max(enhanced_function) * 1.5
            
        # Apply light noise for robustness
        noise_level = 0.02
        noisy_function = enhanced_function + np.random.normal(0, noise_level, len(enhanced_function))
        noisy_function = np.maximum(noisy_function, 0)
        
        # Smooth the function with Savitzky-Golay filter
        window_size = max(1, n_steps // 200)
        if window_size % 2 == 0:
            window_size += 1
        if window_size > 1:
            try:
                smoothed_function = signal.savgol_filter(noisy_function, window_size, 1)
                smoothed_function = np.maximum(smoothed_function, 0)
                noisy_function = smoothed_function
            except Exception:
                pass  # Fall back to original if filtering fails
                
        return noisy_function
        
    def _identify_peaks(self, func: np.ndarray) -> List[Tuple[int, float]]:
        """Identify significant peaks in the function."""
        peaks = []
        for i in range(1, len(func)-1):
            if func[i] > func[i-1] and func[i] > func[i+1]:
                peaks.append((i, func[i]))
        return sorted(peaks, key=lambda x: x[1], reverse=True)
        
    def _optimize_peaks(self, initial_func: np.ndarray, n_steps: int) -> np.ndarray:
        """Optimize peak parameters using differential evolution."""
        try:
            x = np.linspace(-0.25, 0.25, n_steps)
            
            # Early exit if not enough time
            if self._time_remaining() < 5.0:
                return initial_func
            
            # Identify approximate peak locations
            peaks = self._identify_peaks(initial_func)
            
            # Take top peaks for optimization
            if len(peaks) > 0:
                selected_peaks = peaks[:min(8, len(peaks))]
                
                # Refine only peak positions and heights
                def objective(params):
                    # Check time remaining
                    if self._time_remaining() < 2.0:
                        return 1e10  # Large penalty if out of time
                    
                    temp_func = np.zeros_like(x)
                    param_idx = 0
                    for i, (pos_idx, height) in enumerate(selected_peaks):
                        center_pos = x[pos_idx] + (params[param_idx] - 0.5) * 0.05
                        peak_height = height * (1.0 + params[param_idx + 1] * 0.5)
                        width = np.random.uniform(0.02, 0.06)
                        temp_func += peak_height * np.exp(-0.5 * ((x - center_pos) / width)**2)
                        param_idx += 2
                        
                    # Return negative C2 for minimization
                    c2_value = self._compute_c2(temp_func)
                    return -c2_value if c2_value > 0 else 1e10
                    
                # Initial parameter guess (position shift and height multiplier)
                params0 = [0.0] * (len(selected_peaks) * 2)
                
                # Optimize with fewer iterations for speed and time safety
                maxiter = min(20, int(self._time_remaining() * 2))
                if maxiter < 5:
                    return initial_func
                    
                result = differential_evolution(
                    objective,
                    bounds=[(-0.5, 0.5)] * (len(selected_peaks) * 2),
                    maxiter=maxiter,
                    popsize=min(8, maxiter + 2),
                    seed=self.seed,
                    polish=True
                )
                
                if result.success:
                    optimized_params = result.x
                    
                    # Apply optimization results
                    final_func = np.zeros_like(x)
                    param_idx = 0
                    for i, (pos_idx, height) in enumerate(selected_peaks):
                        center_pos = x[pos_idx] + (optimized_params[param_idx] - 0.5) * 0.05
                        peak_height = height * (1.0 + optimized_params[param_idx + 1] * 0.5)
                        width = np.random.uniform(0.02, 0.06)
                        final_func += peak_height * np.exp(-0.5 * ((x - center_pos) / width)**2)
                        param_idx += 2
                        
                    # Add remaining components from original function
                    for i in range(len(initial_func)):
                        if not any(abs(x[i] - x[pos_idx]) < 0.01 for _, pos_idx in selected_peaks):
                            final_func[i] += initial_func[i] * 0.5
                            
                    return final_func
                else:
                    return initial_func
            else:
                return initial_func
                
        except Exception:
            return initial_func
            
    def construct_function(self) -> List[float]:
        """Main function to construct step-function with high C2 value."""
        # Early exit if no time left
        if self._time_remaining() < 1.0:
            return [0.5] * 100
            
        # Determine number of steps
        n_steps = np.random.randint(1000, 10000)
        
        # Early exit if not enough time for main computation
        if self._time_remaining() < 5.0:
            return [0.5] * 100
            
        # Generate base function
        base_function = self._generate_base_function(n_steps)
        
        # Enhance function
        enhanced_function = self._enhance_function(base_function, n_steps)
        
        # Apply peak optimization
        optimized_function = self._optimize_peaks(enhanced_function, n_steps)
        
        # Convert to list and clean up
        result = np.maximum(optimized_function, 0).tolist()
        
        # Add slight noise for robustness
        noise_level = 0.01
        noisy_func = np.array(result) + np.random.normal(0, noise_level, len(result))
        noisy_func = np.maximum(noisy_func, 0)
        
        return noisy_func.tolist()

# Main function that wraps the class
def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value using modular approach."""
    optimizer = AutoCorrelationOptimizer()
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")