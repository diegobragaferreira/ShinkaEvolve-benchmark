# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
from scipy.ndimage import gaussian_filter1d
import random
from typing import List, Tuple
import time

class AdaptiveAutoCorrelationOptimizer:
    """
    An optimized class-based approach for maximizing C2 constant through step function construction.
    Implements a three-phase strategy: initialization, refinement, and validation.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.start_time = None
        
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

    def create_gaussian_structure(self, n_steps: int) -> List[float]:
        """
        Creates a structured Gaussian-based function with controlled peak distribution
        """
        # Define domain
        x = np.linspace(-0.25, 0.25, n_steps)
        
        # Determine number of peaks based on function length
        n_peaks = max(3, min(12, n_steps // 150))
        
        # Initialize function
        f_vals = np.zeros(n_steps)
        
        # Generate peak parameters with strategic placement
        peak_positions = []
        peak_widths = []
        peak_heights = []
        
        # Strategic peak placement with minimum spacing guarantee
        min_spacing = 0.08  # Minimum spacing between peaks (8% of domain)
        
        if n_peaks >= 1:
            # First peak (left side)
            peak_positions.append(np.random.uniform(-0.25, -0.15))
            
        if n_peaks >= 2:
            # Second peak (middle-left)
            peak_positions.append(np.random.uniform(-0.15, -0.05))
            
        if n_peaks >= 3:
            # Third peak (center)
            peak_positions.append(0.0)
            
        if n_peaks >= 4:
            # Fourth peak (middle-right)
            peak_positions.append(np.random.uniform(0.05, 0.15))
            
        if n_peaks >= 5:
            # Fifth peak (right side)
            peak_positions.append(np.random.uniform(0.15, 0.25))
            
        # Adjust for cases where we have more peaks than hardcoded positions
        while len(peak_positions) < n_peaks:
            # Add additional peaks with spacing constraints
            last_pos = peak_positions[-1]
            # Add peak with minimum spacing
            new_pos = last_pos + min_spacing + np.random.uniform(0, 0.05)
            if new_pos < 0.25:
                peak_positions.append(new_pos)
            else:
                # If we exceed boundary, place at boundary
                peak_positions.append(0.25 - min_spacing)
                
        # Generate peak parameters with appropriate scaling
        for pos in peak_positions:
            width = np.random.uniform(0.008, 0.025)
            peak_widths.append(width)
            
            # Height inversely proportional to width for better structural balance
            height = np.random.uniform(0.8, 1.8)
            peak_heights.append(height)
        
        # Create Gaussian curves and sum them
        for center, width, height in zip(peak_positions, peak_widths, peak_heights):
            gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
            f_vals += gaussian
            
        # Apply mathematical principled smoothing with Gaussian kernel
        if n_steps > 50:
            f_vals = gaussian_filter1d(f_vals, sigma=min(2.0, n_steps/100.0))
            
        # Ensure non-negativity
        f_vals = np.maximum(f_vals, 0)
        
        # Normalize appropriately
        if np.max(f_vals) > 0:
            f_vals = f_vals / np.max(f_vals) * 1.2
            
        return f_vals.tolist()

    def refine_with_local_search(self, initial_func: List[float]) -> List[float]:
        """Apply targeted local refinement to improve function quality"""
        try:
            # Work with numpy for better performance
            current_func = np.array(initial_func)
            
            # Apply smoothing and normalization
            smoothed = gaussian_filter1d(current_func, sigma=0.8)
            smoothed = np.maximum(smoothed, 0)
            
            # Reduce extreme values to prevent autoconvolution peak dominance
            # This helps balance ||g||₂² against ||g||₁·||g||∞
            adjusted = smoothed.copy()
            for i in range(len(adjusted)):
                if adjusted[i] > 0.6:
                    adjusted[i] *= 0.92
                    
            return adjusted.tolist()
            
        except Exception:
            return initial_func

    def adaptive_sampling_refinement(self, f_values: List[float], n_steps: int) -> List[float]:
        """
        Apply selective differential evolution with adaptive sampling for optimization
        """
        try:
            # Work with numpy for efficient operations
            original_func = np.array(f_values)
            
            # Determine sampling size based on function size and remaining time
            base_sample_size = min(200, n_steps // 2)
            sample_size = min(base_sample_size, 500)
            
            # Sample indices strategically (focus on high-value regions)
            if len(original_func) > 100:
                # Identify high-value regions to sample more intensively
                high_region_threshold = np.percentile(original_func, 80)
                high_indices = [i for i, val in enumerate(original_func) if val > high_region_threshold]
                
                # Select from high-value regions with some randomization
                if len(high_indices) > 0:
                    selected_indices = random.sample(high_indices, 
                                                   min(len(high_indices), sample_size // 3))
                else:
                    selected_indices = []
                
                # Fill remaining with random samples
                remaining_needed = sample_size - len(selected_indices)
                if remaining_needed > 0:
                    all_indices = list(range(len(original_func)))
                    random_indices = random.sample(all_indices, 
                                                 min(remaining_needed, len(all_indices)))
                    selected_indices.extend(random_indices)
            else:
                # For small functions, sample randomly
                selected_indices = list(range(min(sample_size, len(original_func))))
            
            # Remove duplicates and sort
            selected_indices = sorted(list(set(selected_indices))[:sample_size])
            
            # Prepare objective function for differential evolution
            def objective(params):
                # Create temporary function with updated parameters
                temp_func = original_func.copy()
                for i, idx in enumerate(selected_indices):
                    if i < len(params):
                        temp_func[idx] = max(0.0, params[i])
                
                try:
                    c2_val = self.compute_c2(temp_func.tolist())
                    return -c2_val  # Negative because we want to maximize
                except Exception:
                    return 1e10
                    
            # Set bounds for parameters
            bounds = [(0.0, 3.0) for _ in range(len(selected_indices))]
            
            # Adaptive algorithm parameters based on function size
            max_iter = min(30, max(10, n_steps // 100))
            pop_size = min(8, max(3, n_steps // 200))
            
            # Perform differential evolution
            result = differential_evolution(
                objective,
                bounds,
                maxiter=max_iter,
                popsize=pop_size,
                seed=self.seed,
                disp=False
            )
            
            if result.success and len(result.x) >= len(selected_indices):
                # Update function with optimized values
                final_func = original_func.copy()
                for i, idx in enumerate(selected_indices):
                    if i < len(result.x):
                        final_func[idx] = max(0.0, result.x[i])
                
                return final_func.tolist()
                
        except Exception:
            # Return original if optimization fails
            return f_values
            
        return f_values

    def validate_and_improve(self, func: List[float]) -> List[float]:
        """Validate current function and attempt improvements if needed"""
        try:
            c2_score = self.compute_c2(func)
            
            # If score is very poor, recreate with better structure
            if c2_score < 0.15:
                # Reinitialize with better parameters
                n_steps = len(func)
                return self.create_gaussian_structure(n_steps)
                
            elif c2_score < 0.4:  # Moderate score, try refinement
                # Apply local search
                refined = self.refine_with_local_search(func)
                refined_score = self.compute_c2(refined)
                
                # Accept refinement if it improves score
                if refined_score > c2_score:
                    return refined
                    
        except Exception:
            # Last resort fallback
            n_steps = len(func)
            return [0.5] * n_steps
            
        return func

    def construct_function(self) -> List[float]:
        """Main function to construct step-function with high C2 value."""
        self.start_time = time.time()
        
        # Determine number of steps based on time constraints and randomness
        n_steps = min(8000, max(200, 500 + np.random.randint(0, 1000)))
        
        # Phase 1: Initialization with structured function
        if self._time_remaining():
            initial_function = self.create_gaussian_structure(n_steps)
        else:
            return [0.5] * n_steps
            
        # Phase 2: Local refinement
        if self._time_remaining():
            refined_function = self.refine_with_local_search(initial_function)
        else:
            return initial_function
            
        # Phase 3: Adaptive differential evolution refinement
        if self._time_remaining():
            final_function = self.adaptive_sampling_refinement(refined_function, n_steps)
        else:
            final_function = refined_function
            
        # Phase 4: Validation and final improvement
        if self._time_remaining():
            final_function = self.validate_and_improve(final_function)
            
        return final_function

    def _time_remaining(self, threshold: float = 0.85) -> bool:
        """Check if enough time remains for another phase"""
        if self.start_time is None:
            return True
        elapsed = time.time() - self.start_time
        return elapsed < 90 * threshold

def construct_function() -> List[float]:
    """Wrapper function that maintains interface compatibility"""
    optimizer = AdaptiveAutoCorrelationOptimizer()
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")