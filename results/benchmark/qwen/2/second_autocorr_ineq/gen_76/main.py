# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from numba import jit
import time
import warnings
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class AutoconvolutionNorms:
    """Container for autoconvolution norms"""
    norm_2_squared: float
    norm_1: float
    norm_inf: float
    
    @property
    def c2(self) -> float:
        """Compute C2 value"""
        if self.norm_1 <= 1e-15 or self.norm_inf <= 1e-15:
            return 0.0
        return self.norm_2_squared / (self.norm_1 * self.norm_inf)

@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: list[float]) -> AutoconvolutionNorms:
    """
    Fast computation of the three norms needed for C2 calculation using piecewise linear integration.
    """
    # Convert to numpy array for easier manipulation
    f = np.array(f_values)
    n_steps = len(f)
    
    if n_steps == 0:
        return AutoconvolutionNorms(0.0, 0.0, 0.0)

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

    return AutoconvolutionNorms(norm_2_squared, norm_1, norm_inf)

class FunctionBuilder:
    """Handles the construction of candidate step functions"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
    
    def build_sparse_convex_function(self, n_steps: int) -> np.ndarray:
        """
        Builds function using sparse convex optimization approach with structured basis
        """
        x_domain = np.linspace(-0.25, 0.25, n_steps)
        
        # Create structured dictionary of basis functions
        basis_functions = []
        
        # Center spike basis
        center_spike = np.exp(-8 * x_domain**2)
        basis_functions.append(center_spike)
        
        # Multiple Gaussian-like basis functions with different widths
        widths = [0.05, 0.1, 0.15, 0.2]
        for w in widths:
            gauss = np.exp(-0.5 * (x_domain/w)**2)
            basis_functions.append(gauss)
        
        # Polynomial basis for smooth variations
        poly_basis = [x_domain**i for i in range(1, 4)]
        basis_functions.extend(poly_basis)
        
        # Combine all basis functions
        dictionary = np.column_stack(basis_functions)
        
        # Normalize dictionary columns
        dict_norms = np.linalg.norm(dictionary, axis=0)
        dict_norms[dict_norms == 0] = 1
        dictionary = dictionary / dict_norms
        
        # Create target signal that promotes flat autoconvolution profiles
        target_signal = np.ones_like(x_domain)
        
        # Simple heuristic approach: use first few basis functions with weights
        # This mimics sparse optimization with simple selection
        try:
            # Select first 3 strongest basis functions
            selected_indices = [0, 1, 2]  # Center spike, width 0.05, width 0.1
            
            # Build function from selected components with randomized weights
            f = np.zeros(n_steps)
            for idx in selected_indices:
                if idx < len(basis_functions):
                    weight = np.random.uniform(0.5, 2.0)
                    f += np.maximum(basis_functions[idx], 0) * weight
            
            # Add small random noise to break symmetry  
            noise = np.random.normal(0, 0.01, n_steps)
            f += noise
            
            # Enforce non-negativity and normalization
            f = np.maximum(f, 0)
            if np.sum(f) > 0:
                f = f / np.sum(f) * 10
                
            return f
            
        except Exception:
            pass
        
        # Fallback method
        return self._fallback_function(n_steps, x_domain)
    
    def _fallback_function(self, n_steps: int, x_domain: np.ndarray) -> np.ndarray:
        """Fallback function construction when sparse method fails"""
        f = np.zeros(n_steps)
        
        # Add a central peak with exponential decay
        f += np.exp(-10 * x_domain**2)
        
        # Add some sinusoidal modulation to avoid degeneracy
        f += 0.3 * np.sin(10 * np.pi * x_domain) * np.exp(-x_domain**2/0.1)
        
        # Add some noise for robustness
        f += np.random.normal(0, 0.05, n_steps)
        
        # Enforce non-negativity and normalization
        f = np.maximum(f, 0)
        if np.sum(f) > 0:
            f = f / np.sum(f) * 10
        
        return f

class Optimizer:
    """Handles the optimization process and finding of best function"""
    
    def __init__(self, builder: FunctionBuilder, max_time: int = 85):
        self.builder = builder
        self.max_time = max_time
        self.best_c2 = -1
        self.best_function = None
        self.start_time = None
    
    def run_optimization(self) -> List[float]:
        """Main optimization loop"""
        self.start_time = time.time()
        max_attempts = 30
        
        for attempt in range(max_attempts):
            # Check time budget
            if time.time() - self.start_time > self.max_time:
                break
                
            # Try different number of steps to find optimal
            n_steps = np.random.randint(1500, 4000)
            
            # Generate function using convex optimization approach
            f_values = self.builder.build_sparse_convex_function(n_steps)
            
            # Evaluate the function
            try:
                norms = compute_autoconvolution_norms_fast(f_values.tolist())
                c2 = norms.c2
                
                # Keep the best function
                if c2 > self.best_c2:
                    self.best_c2 = c2
                    self.best_function = f_values.tolist()
                    
            except Exception:
                # Skip invalid functions
                continue
        
        # Return the best function found, or fallback
        if self.best_function is not None:
            return self.best_function
        else:
            # Fallback to a simpler construction
            n_steps = 1000
            return [1.0] * n_steps

def construct_function() -> List[float]:
    """
    Main function to construct optimized step function using convex optimization approach.
    """
    # Initialize components
    builder = FunctionBuilder(seed=42)
    optimizer = Optimizer(builder, max_time=85)
    
    # Run optimization
    return optimizer.run_optimization()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")