# EVOLVE-BLOCK-START

import numpy as np
from numba import njit
import time

class StepFunctionOptimizer:
    """
    A modular, pipeline-based optimizer for finding step functions that maximize C2.
    """
    
    def __init__(self, n_steps=1000, max_iterations=50, patience=8):
        self.n_steps = n_steps
        self.max_iterations = max_iterations
        self.patience = patience
        self.best_c2 = -float('inf')
        self.best_function = None
        
    @staticmethod
    @njit
    def _compute_convolution_norms(f_values, domain_length=0.5):
        """
        Compute the three norms needed for C2 calculation using JIT compilation.
        """
        n_steps = len(f_values)
        if n_steps == 0:
            return 0.0, 0.0, 0.0

        # Step size
        dx = domain_length / n_steps

        # Compute autoconvolution g = f * f 
        g_size = 2 * n_steps - 1
        g = np.zeros(g_size)

        # Compute autoconvolution with proper dx scaling
        for i in range(n_steps):
            for j in range(n_steps):
                k = i + j
                if 0 <= k < g_size:
                    g[k] += f_values[i] * f_values[j] * dx

        # Compute norms using trapezoidal rule for ||g||₂²
        g2_sq = 0.0
        for i in range(len(g)-1):
            g2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)

        # ||g||₁ = sum(|g_i| * dx)  
        g1 = np.sum(np.abs(g)) * dx

        # ||g||∞ = max(|g_i|)
        ginf = np.max(np.abs(g))

        return g2_sq, g1, ginf
    
    @staticmethod
    @njit
    def _compute_c2(f_values):
        """Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞)"""
        g2_sq, g1, ginf = StepFunctionOptimizer._compute_convolution_norms(f_values)

        if g1 <= 1e-15 or ginf <= 1e-15:
            return 0.0

        return g2_sq / (g1 * ginf)
    
    @staticmethod
    def _initialize_function(n_steps):
        """
        Create an initial step function with mathematical structure.
        """
        # Create symmetric bump pattern that tends to produce flat autoconvolutions
        half = n_steps // 2
        f_values = np.zeros(n_steps)
        
        for i in range(n_steps):
            if i < half:
                # Increasing ramp
                f_values[i] = 1.0 * (i / half)
            else:
                # Decreasing ramp  
                f_values[i] = 1.0 * ((n_steps - i) / half)
        
        # Normalize to reasonable values
        total_area = np.sum(f_values) * (0.5 / n_steps)
        if total_area > 0:
            f_values = f_values / total_area
            
        return f_values.tolist()
    
    @staticmethod
    def _smooth_function(f_values, smoothing_factor=0.3):
        """
        Apply smoothing to reduce sensitivity to small perturbations.
        """
        if len(f_values) <= 1:
            return f_values
            
        smoothed = []
        for i in range(len(f_values)):
            # Simple averaging with neighbors
            left = max(0, i - 1)
            right = min(len(f_values), i + 2)
            avg = sum(f_values[left:right]) / (right - left)
            smoothed.append(avg)
            
        return smoothed
    
    def _local_search_refinement(self, f_values):
        """
        Perform adaptive local search to refine the function.
        """
        if len(f_values) == 0:
            return f_values
            
        best_f = np.array(f_values, dtype=np.float64)
        best_c2 = self._compute_c2(best_f)
        
        recent_improvements = []
        current_patience = 0
        
        # Adaptive parameters
        num_modifications_base = max(1, min(20, len(f_values) // 15))
        improvement_threshold = 0.001
        
        for iteration in range(self.max_iterations):
            test_f = best_f.copy()
            
            # Adaptively adjust number of modifications
            num_modifications = num_modifications_base
            if len(recent_improvements) > 5:
                avg_improvement = np.mean(recent_improvements[-5:])
                if avg_improvement < improvement_threshold * 0.1:
                    num_modifications = max(1, num_modifications // 2)
            
            # Choose random indices to modify
            mod_indices = np.random.choice(len(test_f), num_modifications, replace=False)
            
            # Apply modifications
            for idx in mod_indices:
                change = np.random.normal(0, 0.05 * best_f[idx])
                test_f[idx] = max(0, best_f[idx] + change)
            
            # Evaluate improvement
            test_c2 = self._compute_c2(test_f)
            improvement = test_c2 - best_c2
            
            if test_c2 > best_c2:
                best_c2 = test_c2
                best_f = test_f
                recent_improvements.append(improvement)
                current_patience = 0
            else:
                current_patience += 1
                recent_improvements.append(improvement)
            
            # Early stopping
            if current_patience >= self.patience:
                break
                
        return best_f.tolist()
    
    def optimize(self):
        """
        Main optimization pipeline.
        """
        # Stage 1: Initialize function
        initial_function = self._initialize_function(self.n_steps)
        
        # Stage 2: Smooth to reduce sensitivity
        smoothed_function = self._smooth_function(initial_function)
        
        # Stage 3: Local search refinement
        refined_function = self._local_search_refinement(smoothed_function)
        
        # Final evaluation
        final_c2 = self._compute_c2(refined_function)
        
        return refined_function, final_c2

def construct_function() -> list[float]:
    """
    Main entry point that constructs a step-function with high C2 value.
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    
    # Create optimizer with tuned parameters
    optimizer = StepFunctionOptimizer(
        n_steps=1000,
        max_iterations=50,
        patience=8
    )
    
    # Run optimization
    best_function, best_c2 = optimizer.optimize()
    
    # Store best result for reference
    if best_c2 > optimizer.best_c2:
        optimizer.best_c2 = best_c2
        optimizer.best_function = best_function
        
    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")