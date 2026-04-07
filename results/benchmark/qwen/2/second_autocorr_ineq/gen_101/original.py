# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List, Tuple, Any

class StepFunctionOptimizer:
    """Modular optimizer for constructing step functions to maximize C2 constant."""
    
    def __init__(self, seed: int = 42):
        """Initialize optimizer with deterministic behavior."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_base_function(self, n_steps: int) -> np.ndarray:
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
        
    def enhance_function(self, base_function: np.ndarray, n_steps: int) -> np.ndarray:
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
        
        # Smooth the function
        window_size = max(1, n_steps // 200)
        if window_size % 2 == 0:
            window_size += 1
        if window_size > 1:
            smoothed_function = signal.savgol_filter(noisy_function, window_size, 1)
            smoothed_function = np.maximum(smoothed_function, 0)
            noisy_function = smoothed_function
            
        return noisy_function
        
    def compute_c2(self, func: List[float]) -> float:
        """Compute C2 value for given function."""
        # Convert to numpy array if needed
        if isinstance(func, list):
            func = np.array(func)
            
        # Compute autoconvolution g = f * f
        g = np.convolve(func, func, mode='full')
        g = g[len(g)//2:]
        
        # Truncate if necessary
        if len(g) > len(func):
            g = g[:len(func)]
            
        # Compute norms
        norm_2_sq = np.sum(g**2) * (0.5 / len(func))
        norm_1 = np.sum(np.abs(g)) / (len(g) + 1)
        norm_inf = np.max(np.abs(g))
        
        if norm_1 == 0 or norm_inf == 0:
            return 0.0
            
        return norm_2_sq / (norm_1 * norm_inf)
        
    def optimize_peaks(self, initial_func: List[float], n_steps: int) -> List[float]:
        """Optimize peak parameters using differential evolution."""
        try:
            x = np.linspace(-0.25, 0.25, n_steps)
            
            # Identify approximate peak locations
            peaks = []
            for i in range(1, len(initial_func)-1):
                if initial_func[i] > initial_func[i-1] and initial_func[i] > initial_func[i+1]:
                    peaks.append((i, initial_func[i]))
                    
            # Take top peaks
            if len(peaks) > 0:
                peaks.sort(key=lambda x: x[1], reverse=True)
                selected_peaks = peaks[:min(8, len(peaks))]
                
                # Refine only peak positions and heights
                def objective(params):
                    temp_func = np.zeros_like(x)
                    for i, (pos_idx, height) in enumerate(selected_peaks):
                        center_pos = x[pos_idx] + (params[i*2] - 0.5) * 0.05
                        peak_height = height * (1.0 + params[i*2+1] * 0.5)
                        width = np.random.uniform(0.02, 0.06)
                        temp_func += peak_height * np.exp(-0.5 * ((x - center_pos) / width)**2)
                    return -self.compute_c2(temp_func)
                    
                # Initial parameter guess
                params0 = [0.0] * (len(selected_peaks) * 2)
                
                # Optimize with fewer iterations for speed
                result = differential_evolution(
                    objective,
                    bounds=[(-0.5, 0.5)] * (len(selected_peaks) * 2),
                    maxiter=30,
                    popsize=8,
                    seed=self.seed,
                    polish=True
                )
                optimized_params = result.x
                
                # Apply optimization results
                final_func = np.zeros_like(x)
                for i, (pos_idx, height) in enumerate(selected_peaks):
                    center_pos = x[pos_idx] + (optimized_params[i*2] - 0.5) * 0.05
                    peak_height = height * (1.0 + optimized_params[i*2+1] * 0.5)
                    width = np.random.uniform(0.02, 0.06)
                    final_func += peak_height * np.exp(-0.5 * ((x - center_pos) / width)**2)
                    
                # Add remaining peaks from original
                for i in range(len(initial_func)):
                    if not any(abs(x[i] - x[pos_idx]) < 0.01 for _, pos_idx in selected_peaks):
                        final_func[i] += initial_func[i] * 0.5
                        
                return final_func.tolist()
            else:
                return initial_func
                
        except Exception:
            return initial_func
            
    def construct_function(self) -> List[float]:
        """Main function to construct step-function with high C2 value."""
        # Determine number of steps
        n_steps = np.random.randint(1000, 10000)
        
        # Generate base function
        base_function = self.generate_base_function(n_steps)
        
        # Enhance function
        enhanced_function = self.enhance_function(base_function, n_steps)
        
        # Convert to list
        step_values = enhanced_function.tolist()
        
        # Apply peak optimization
        optimized_step_values = self.optimize_peaks(step_values, n_steps)
        
        # Final cleanup and return
        final_array = np.array(optimized_step_values)
        final_array = np.maximum(final_array, 0)
        
        # Add slight noise for robustness
        noise_level = 0.01
        noisy_func = final_array + np.random.normal(0, noise_level, len(final_array))
        noisy_func = np.maximum(noisy_func, 0)
        
        return noisy_func.tolist()

# Main function that wraps the class
def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value using modular approach."""
    optimizer = StepFunctionOptimizer()
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")