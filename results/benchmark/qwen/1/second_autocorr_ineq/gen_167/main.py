# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import differential_evolution
from scipy.signal import convolve
import warnings
from typing import Tuple

class FourierDomainOptimizer:
    """Optimizes step functions in the frequency domain for better numerical properties."""
    
    def __init__(self, n_steps: int = 1000, max_time_seconds: float = 90.0):
        self.n_steps = n_steps
        self.max_time_seconds = max_time_seconds
        self.domain = [-0.25, 0.25]
        self.step_width = (self.domain[1] - self.domain[0]) / n_steps
        
    def _construct_time_domain_from_fourier(self, fourier_coeffs: np.ndarray) -> np.ndarray:
        """
        Convert frequency domain representation to time domain step function
        using inverse FFT. Ensures non-negativity and proper scaling.
        """
        # Use real FFT for real-valued results
        # Make sure we have the right number of coefficients
        n_coeffs = len(fourier_coeffs)
        
        # If odd number, make it even for symmetric real FFT
        if n_coeffs % 2 == 1:
            fourier_coeffs = np.append(fourier_coeffs, 0)
            n_coeffs = len(fourier_coeffs)
            
        # Generate time domain signal using inverse FFT
        # Note: we're working with a periodic extension
        time_signal = ifft(fourier_coeffs).real
        
        # Periodic extension - take the first n_steps samples
        if len(time_signal) >= self.n_steps:
            time_signal = time_signal[:self.n_steps]
        else:
            # Pad with zeros if needed
            time_signal = np.pad(time_signal, (0, self.n_steps - len(time_signal)), 'constant')
        
        # Ensure non-negativity with soft clipping
        # Use a smooth sigmoid-like transformation to preserve gradients
        time_signal = np.maximum(time_signal, 0)
        
        return time_signal
    
    def _compute_autoconvolution_norms(self, f_values: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute the three norms needed for C2 calculation using efficient methods.
        """
        # Ensure non-negative values
        f = np.maximum(f_values, 0.0)
        
        # Compute autoconvolution g = f * f (discrete convolution)
        g = convolve(f, f, mode='full')
        
        # Extract the central portion that represents the main interval
        n = len(f)
        middle_idx = n - 1
        half_width = n
        
        # Take the central part of the convolution
        g_centered = g[middle_idx - half_width + 1 : middle_idx + half_width]
        
        # Compute the norms
        g_squared = g_centered ** 2
        g_abs = np.abs(g_centered)
        
        # ||g||₂² - sum of squares (trapezoidal integration approximation)
        # For piecewise linear integration: each interval contributes (h/3)(y1² + y1y2 + y2²) 
        norm_g2_sq = 0.0
        if len(g_centered) >= 2:
            for i in range(len(g_centered) - 1):
                y1 = g_centered[i]
                y2 = g_centered[i+1]
                norm_g2_sq += (y1*y1 + y1*y2 + y2*y2) / 3.0
        
        # ||g||₁ - sum of absolute values
        norm_g1 = np.sum(g_abs)
        
        # ||g||∞ - maximum absolute value
        norm_ginf = np.max(g_abs)
        
        return norm_g2_sq, norm_g1, norm_ginf
    
    def compute_c2_from_fourier(self, fourier_coeffs: np.ndarray) -> float:
        """
        Compute C2 value directly from frequency domain representation.
        """
        try:
            # Convert to time domain
            f_values = self._construct_time_domain_from_fourier(fourier_coeffs)
            
            # Compute norms
            norm_g2_sq, norm_g1, norm_ginf = self._compute_autoconvolution_norms(f_values)
            
            # Avoid division by zero
            if norm_g1 < 1e-15 or norm_ginf < 1e-15:
                return 0.0
            
            # C2 = ||g||₂² / (||g||₁ · ||g||∞)
            c2 = norm_g2_sq / (norm_g1 * norm_ginf)
            
            return c2
            
        except Exception as e:
            warnings.warn(f"Error in C2 computation: {str(e)}")
            return 0.0
    
    def _initialize_fourier_coefficients(self) -> np.ndarray:
        """
        Initialize frequency domain coefficients with mathematically informed patterns.
        """
        # Create a mix of low-frequency components (smooth) and high-frequency (structured)
        # For a 1D signal of length n, we have n/2+1 unique frequency components for real signals
        
        # Generate frequency components with decreasing magnitude
        coeffs = np.zeros(self.n_steps, dtype=complex)
        
        # Base pattern: some low frequency components plus structured noise
        for i in range(self.n_steps // 2):
            # Low frequency components (more energy)
            if i < 5:
                coeffs[i] = complex(1.0 + np.random.random() * 0.5, 0.0)  # Real coefficients
            elif i < 20:
                # Medium frequencies with decay
                coeffs[i] = complex(0.5 + np.random.random() * 0.3, 0.0)  # Real coefficients
            else:
                # High frequencies - reduce energy
                coeffs[i] = complex(0.1 + np.random.random() * 0.1, 0.0)  # Real coefficients
                
        # Mirror for negative frequencies (for real signals)
        if self.n_steps % 2 == 0:  # Even number of samples
            coeffs[self.n_steps // 2] = complex(np.random.random() * 0.5, 0.0)
            for i in range(1, self.n_steps // 2):
                coeffs[self.n_steps - i] = np.conj(coeffs[i])
        else:  # Odd number of samples
            coeffs[self.n_steps // 2] = complex(np.random.random() * 0.5, 0.0)
            for i in range(1, (self.n_steps + 1) // 2):
                coeffs[self.n_steps - i] = np.conj(coeffs[i])
        
        # Convert to real values for easier optimization
        return np.real(coeffs)
    
    def _adaptive_evolutionary_search(self, initial_coeffs: np.ndarray) -> np.ndarray:
        """
        Perform evolutionary optimization in frequency domain with adaptive strategies.
        """
        start_time = time.time()
        
        # Convert complex coefficients to real parameters for optimization
        n_coeffs = len(initial_coeffs)
        bounds = [(0.0, 3.0) for _ in range(n_coeffs)]
        
        def objective(x):
            # Convert real vector back to complex frequency domain representation
            # For simplicity, we'll use a mapping that keeps the structure
            fourier_coeffs = x.astype(complex)
            
            # Ensure non-negative values
            c2 = self.compute_c2_from_fourier(fourier_coeffs)
            return -c2  # Minimize negative to maximize C2
            
        # Multi-phase optimization
        try:
            # Phase 1: Coarse-grained optimization
            coarse_bounds = [(0.0, 3.0) for _ in range(max(10, n_coeffs // 10))]
            coarse_result = differential_evolution(
                objective,
                bounds,
                maxiter=min(50, int(self.max_time_seconds * 0.4)),
                popsize=max(5, n_coeffs // 20),
                seed=42,
                disp=False
            )
            
            best_solution = coarse_result.x if coarse_result.success else initial_coeffs
            
        except Exception:
            best_solution = initial_coeffs
            
        # Phase 2: Fine-grained optimization (local refinement)
        try:
            # Apply local refinement only if enough time remains
            if time.time() - start_time < self.max_time_seconds * 0.8:
                # Convert back to complex for fine tuning
                fine_coeffs = best_solution.astype(complex)
                
                # Perform additional local refinement
                # Note: We can do this because we're working in a continuous domain
                # but the actual computation will remain the same
                
                refinement_result = differential_evolution(
                    objective,
                    bounds,
                    x0=best_solution,
                    maxiter=min(30, int(self.max_time_seconds * 0.3)),
                    popsize=max(5, n_coeffs // 15),
                    seed=42,
                    disp=False
                )
                
                if refinement_result.success:
                    final_solution = refinement_result.x
                else:
                    final_solution = best_solution
                    
            else:
                final_solution = best_solution
                
        except Exception:
            final_solution = best_solution
            
        # Ensure final solution is valid
        final_solution = np.clip(final_solution, 0.0, None)
        
        return final_solution

def construct_function() -> list[float]:
    """
    Main entry point for constructing step-function with high C2 value.
    Uses frequency domain optimization approach.
    """
    # Set up the optimizer with appropriate parameters
    start_time = time.time()
    
    # Use a moderate number of steps for reasonable computational budget
    n_steps = 800  # Good compromise between resolution and time
    
    # Create optimizer
    optimizer = FourierDomainOptimizer(n_steps=n_steps, max_time_seconds=90.0)
    
    try:
        # Initialize with mathematically informed pattern
        initial_coeffs = optimizer._initialize_fourier_coefficients()
        
        # Perform adaptive evolutionary search in frequency domain
        optimized_coeffs = optimizer._adaptive_evolutionary_search(initial_coeffs)
        
        # Convert back to time domain for final result
        final_function = optimizer._construct_time_domain_from_fourier(optimized_coeffs)
        
        # Ensure valid output format
        result = final_function.tolist()
        
    except Exception as e:
        # Fallback to simple initialization if optimization fails
        warnings.warn(f"Optimization failed with error: {e}. Using fallback.")
        # Generate simple pattern
        n_steps = 500
        result = [0.5] * n_steps
        
    # Final check to ensure non-negativity and proper format
    result = [max(0, val) for val in result]
    
    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")