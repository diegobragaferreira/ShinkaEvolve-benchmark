# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
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
    
    def create_structured_gaussian_individual(self, n_steps: int, n_peaks: int = None) -> List[float]:
        """
        Create a structured individual using adaptive Gaussian peak construction
        with controlled spacing and amplitude scaling.
        """
        # Determine number of peaks based on function length
        if n_peaks is None:
            n_peaks = max(3, min(15, n_steps // 100))

        # Create step function with Gaussian peaks
        x = np.linspace(-0.25, 0.25, n_steps)
        f_vals = np.zeros(n_steps)

        # Generate peak parameters with controlled spacing
        peak_positions = []
        peak_widths = []
        peak_heights = []

        # Distribute peaks with logarithmic spacing for better diversity
        for i in range(n_peaks):
            if i == 0:
                # First peak near left edge
                pos = np.random.uniform(-0.25, -0.1)
            elif i == n_peaks - 1:
                # Last peak near right edge
                pos = np.random.uniform(0.1, 0.25)
            else:
                # Intermediate peaks with controlled spacing
                log_min = np.log(0.05)
                log_max = np.log(0.45)
                log_pos = np.random.uniform(log_min, log_max)
                rel_pos = np.exp(log_pos)
                pos = -0.25 + rel_pos * 0.5

            peak_positions.append(pos)
            # Width inversely related to height for better control
            width = np.random.uniform(0.005, 0.02)
            peak_widths.append(width)
            # Height inversely proportional to width to maintain balance
            height = np.random.uniform(0.5, 2.0)
            peak_heights.append(height)

        # Create Gaussian curves for each peak
        for center, width, height in zip(peak_positions, peak_widths, peak_heights):
            gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
            f_vals += gaussian

        # Apply smoothing to reduce extreme variations
        if n_steps > 50:
            f_vals = signal.savgol_filter(f_vals, min(51, n_steps-1), 3)

        # Ensure non-negativity
        f_vals = np.maximum(f_vals, 0)

        # Normalize to reasonable range
        if np.max(f_vals) > 0:
            f_vals = f_vals / np.max(f_vals) * 1.5

        return f_vals.tolist()
    
    def create_multi_peak_distribution(self, n_steps: int) -> List[float]:
        """Create a multi-peak distribution inspired by successful patterns"""
        # Create a bell-shaped distribution with multiple peaks
        x = np.linspace(-0.25, 0.25, n_steps)
        
        # Multiple Gaussian components with different characteristics
        # This creates a structure that typically has good autoconvolution properties
        base_shape = (
            0.4 * np.exp(-x**2 / 0.02) + 
            0.3 * np.exp(-((x - 0.1)**2) / 0.01) + 
            0.2 * np.exp(-((x + 0.1)**2) / 0.01) +
            0.1 * np.exp(-((x - 0.05)**2) / 0.005)
        )
        
        # Normalize and add controlled noise
        base_shape = base_shape / np.max(base_shape) * 0.8
        noise = np.random.normal(0, 0.02, n_steps)
        final_shape = np.maximum(base_shape + noise, 0)
        
        return final_shape.tolist()
    
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
    
    def selective_differential_evolution(self, initial_func: List[float], n_steps: int) -> List[float]:
        """Apply targeted differential evolution on key parameters"""
        try:
            # Use a more efficient approach - target only peak parameters
            # Extract approximate peak locations using gradient-based detection
            func_array = np.array(initial_func)
            
            # Find potential peak locations (local maxima)
            # Simple approach: look at points with high values and local maxima
            threshold = np.percentile(func_array, 70)
            peak_candidates = []
            
            for i in range(1, len(func_array) - 1):
                if (func_array[i] > func_array[i-1] and 
                    func_array[i] > func_array[i+1] and 
                    func_array[i] > threshold):
                    peak_candidates.append(i)
            
            # If we have few candidates, expand search
            if len(peak_candidates) < 3:
                # Add some additional points with high values
                high_points = np.where(func_array > threshold)[0].tolist()
                peak_candidates.extend(high_points[:10])
                peak_candidates = list(set(peak_candidates))[:15]
            
            # Use only the most promising peak locations for optimization
            selected_indices = sorted(peak_candidates[:min(20, len(peak_candidates))])
            
            if len(selected_indices) < 2:
                # Fallback to random selection if inadequate peaks detected
                selected_indices = sorted(random.sample(range(n_steps), min(10, n_steps)))
            
            # Optimze only the selected indices
            def objective(params):
                # Create new function with modified peak parameters
                temp_func = func_array.copy()
                
                # Update the selected indices with new parameters
                for i, idx in enumerate(selected_indices):
                    if i < len(params):
                        temp_func[idx] = max(0, params[i])
                
                try:
                    c2_val = self.compute_c2(temp_func.tolist())
                    return -c2_val  # Negative because we want to maximize
                except Exception:
                    return 1e10
            
            # Set bounds for parameters (0 to 2.0 for peak heights)
            bounds = [(0.0, 2.0) for _ in range(len(selected_indices))]
            
            # Select a subset for efficiency - limit to 50 parameters
            if len(selected_indices) > 50:
                # Sample indices rather than optimizing all
                sample_indices = sorted(random.sample(selected_indices, 50))
                sample_bounds = [bounds[i] for i in range(len(selected_indices)) if selected_indices[i] in sample_indices]
                sample_params = [func_array[i] for i in sample_indices]
            else:
                sample_indices = selected_indices
                sample_bounds = bounds
                sample_params = [func_array[i] for i in selected_indices]
            
            # Perform differential evolution with fewer iterations for speed
            result = differential_evolution(
                objective,
                sample_bounds,
                maxiter=30,  # Reduced iterations for speed
                popsize=5,   # Reduced population size
                seed=self.seed,
                disp=False
            )
            
            # Update the function with optimized values
            final_func = func_array.copy()
            if result.success and len(result.x) >= len(sample_indices):
                for i, idx in enumerate(sample_indices):
                    if i < len(result.x):
                        final_func[idx] = max(0, result.x[i])
            
            return final_func.tolist()
            
        except Exception:
            return initial_func

    def construct_function(self) -> List[float]:
        """Main function to construct step-function with high C2 value."""
        start_time = time.time()

        # Determine number of steps with time budget consideration
        n_steps = min(1000, max(100, 500 + int(np.random.randint(0, 200) * 2)))

        # Phase 1: Create baseline function with good structure
        if time.time() - start_time > 80:  # Leave margin for processing
            return [0.5] * n_steps

        # Start with structured Gaussian approach (better than simple distribution)
        best_function = self.create_structured_gaussian_individual(n_steps)
        
        # Phase 2: Apply local refinement
        if time.time() - start_time > 85:
            return best_function
            
        refined_function = self.improve_with_local_search(best_function)
        
        # Phase 3: Apply selective differential evolution for targeted optimization
        if time.time() - start_time > 87:
            return refined_function
            
        try:
            optimized_function = self.selective_differential_evolution(refined_function, n_steps)
        except Exception:
            optimized_function = refined_function
            
        # Phase 4: Final validation and adjustment
        c2_score = self.compute_c2(optimized_function)
        
        if time.time() - start_time > 88:
            return optimized_function
            
        # Apply one final adjustment if needed
        if c2_score < 0.1:  # Very low score - try alternative
            alternative = self.create_multi_peak_distribution(n_steps)
            alt_c2 = self.compute_c2(alternative)
            if alt_c2 > c2_score:
                optimized_function = alternative

        return optimized_function

def construct_function() -> List[float]:
    """Wrapper function that maintains interface compatibility"""
    optimizer = AutoCorrelationOptimizer()
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")