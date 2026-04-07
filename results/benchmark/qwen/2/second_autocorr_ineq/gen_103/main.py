# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import random
from typing import List, Tuple
import time

class AutoCorrelationOptimizer:
    """Optimized optimizer for maximizing C2 constant through step function construction."""
    
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
        
        # Compute norms efficiently
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
    
    def create_log_spaced_peaks(self, n_peaks: int, domain_width: float = 0.5) -> np.ndarray:
        """Generate logarithmically spaced peak positions to avoid clustering"""
        # Generate logarithmically spaced positions in [0, 1] then map to [-domain_width/2, domain_width/2]
        log_positions = np.logspace(np.log10(0.01), np.log10(1.0), n_peaks, endpoint=True)
        positions = (log_positions - 0.5) * domain_width
        return positions
    
    def generate_multi_scale_gaussian(self, x: np.ndarray, peak_positions: np.ndarray, 
                                    peak_heights: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        """Generate function with multiple Gaussian scales for better structure"""
        result = np.zeros_like(x)
        for pos, height, sigma in zip(peak_positions, peak_heights, sigmas):
            result += height * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
        return result
    
    def create_balanced_distribution(self, n_steps: int) -> List[float]:
        """Create a sophisticated balanced distribution with multi-scale peaks"""
        # Create domain
        x = np.linspace(-0.25, 0.25, n_steps)
        
        # Generate multiple sets of peaks with different scales
        n_peaks_total = max(3, n_steps // 100)
        
        # Primary peak structure
        primary_positions = self.create_log_spaced_peaks(n_peaks_total // 2)
        primary_heights = np.random.exponential(1.0, n_peaks_total // 2) * 0.5
        primary_sigmas = np.full(n_peaks_total // 2, 0.01)
        
        # Secondary peak structure
        secondary_positions = self.create_log_spaced_peaks(n_peaks_total - n_peaks_total // 2, 0.3)
        secondary_heights = np.random.exponential(1.0, n_peaks_total - n_peaks_total // 2) * 0.3
        secondary_sigmas = np.full(n_peaks_total - n_peaks_total // 2, 0.02)
        
        # Combine all peaks
        all_positions = np.concatenate([primary_positions, secondary_positions])
        all_heights = np.concatenate([primary_heights, secondary_heights])
        all_sigmas = np.concatenate([primary_sigmas, secondary_sigmas])
        
        # Generate the combined function
        base_shape = self.generate_multi_scale_gaussian(x, all_positions, all_heights, all_sigmas)
        
        # Normalize and apply final adjustments
        if np.max(base_shape) > 0:
            base_shape = base_shape / np.max(base_shape) * 0.8
        
        # Add small random noise to break symmetry
        noise = np.random.normal(0, 0.01, n_steps)
        final_shape = np.maximum(base_shape + noise, 0)
        
        return final_shape.tolist()
    
    def adaptive_local_improvement(self, f_values: List[float], iterations: int = 3) -> List[float]:
        """Apply adaptive local refinement to improve the function"""
        current_func = np.array(f_values)
        
        for iteration in range(iterations):
            # Apply Gaussian smoothing with adaptive sigma
            sigma = max(0.1, 1.0 / (iteration + 1))
            smoothed = gaussian_filter1d(current_func, sigma=sigma)
            
            # Apply adaptive adjustments based on current behavior
            # If peaks are too dominant, reduce them
            if len(current_func) > 10:
                # Calculate local variance to identify problematic peaks
                window_size = max(3, len(current_func) // 20)
                local_mean = np.convolve(current_func, np.ones(window_size)/window_size, mode='same')
                
                # Reduce high values near local maxima
                adjustment_factor = 0.95
                adjusted = current_func * adjustment_factor
                # But keep the smooth version for the final result
                current_func = np.where(current_func > local_mean * 1.2, 
                                       adjusted, 
                                       smoothed)
            
            current_func = np.maximum(current_func, 0)
            
        return current_func.tolist()
    
    def smart_refinement(self, initial_func: List[float]) -> List[float]:
        """Apply smart refinement to boost performance"""
        refined = initial_func.copy()
        
        # Apply multiple rounds of improvement
        refined = self.adaptive_local_improvement(refined, 2)
        
        # Apply final smoothing
        arr = np.array(refined)
        smoothed = gaussian_filter1d(arr, sigma=0.3)
        smoothed = np.maximum(smoothed, 0)
        
        return smoothed.tolist()
    
    def construct_function(self) -> List[float]:
        """Main function to construct step-function with high C2 value."""
        start_time = time.time()
        
        # Determine number of steps with time budget consideration
        n_steps = min(2000, max(200, 1000 + int(np.random.randint(0, 500) * 2)))
        
        # Phase 1: Create sophisticated initial function
        if time.time() - start_time > 80:  # Leave margin for processing
            return [0.5] * n_steps
            
        try:
            # Create highly structured function with multiple scales
            best_function = self.create_balanced_distribution(n_steps)
            
            # Phase 2: Apply smart refinement
            if time.time() - start_time > 85:
                return best_function
                
            refined_function = self.smart_refinement(best_function)
            
            # Phase 3: Final validation and optional adjustment
            c2_score = self.compute_c2(refined_function)
            
            # If we're close to timeout, just return what we have
            if time.time() - start_time > 88:
                return refined_function
                
            # Apply final check for quality
            if c2_score < 0.1:  # Very low score - try a different approach
                # Try a simpler but well-structured approach
                simple_func = np.ones(n_steps) * 0.5
                simple_func = gaussian_filter1d(simple_func, sigma=0.5)
                simple_func = np.maximum(simple_func, 0)
                alt_c2 = self.compute_c2(simple_func.tolist())
                if alt_c2 > c2_score:
                    refined_function = simple_func.tolist()
                    
        except Exception:
            # Fallback to simple approach
            simple_func = np.ones(n_steps) * 0.5
            simple_func = gaussian_filter1d(simple_func, sigma=0.5)
            simple_func = np.maximum(simple_func, 0)
            refined_function = simple_func.tolist()
        
        return refined_function

def construct_function() -> List[float]:
    """Wrapper function that maintains interface compatibility"""
    optimizer = AutoCorrelationOptimizer()
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")