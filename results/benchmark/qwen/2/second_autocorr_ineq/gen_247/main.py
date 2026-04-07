# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import minimize
import warnings
from itertools import combinations
import time

class SpectralPeakOptimizer:
    """Optimizes step functions by working in spectral domain with Fourier transforms."""
    
    def __init__(self, n_points=2000):
        self.n_points = n_points
        self.domain = np.linspace(-0.25, 0.25, n_points)
        self.dx = self.domain[1] - self.domain[0] if len(self.domain) > 1 else 1.0
        
    def compute_autoconvolution_norms(self, f_values):
        """Compute the three norms needed for C2 calculation with improved numerical stability."""
        if not f_values:
            return 0.0, 1e-12, 1e-12
            
        # Convert to numpy array
        f = np.array(f_values, dtype=np.float64)
        
        # Validate input
        if len(f) < 1:
            return 0.0, 1e-12, 1e-12
            
        # Compute autoconvolution using fast convolution
        g = signal.convolve(f, f, mode='full')
        g = g[len(f)-1:]  # Keep only the relevant part
        
        # Compute norms with numerical stability checks
        g_abs = np.abs(g)
        
        # ||g||₂² (L2 norm squared) - use trapezoidal rule properly
        g_sq = g_abs ** 2
        if len(g_sq) < 2:
            norm_2_sq = 0.0 if len(g_sq) == 0 else g_sq[0]
        else:
            # Trapezoidal integration: sum((y[i] + y[i+1])/2 * delta_x)
            # We approximate dx as 1 since we're dealing with discrete samples
            norm_2_sq = np.sum((g_sq[:-1] + g_sq[1:]) / 2.0)
        
        # ||g||₁ (L1 norm) - sum of absolute values
        norm_1 = np.sum(g_abs)
        
        # ||g||∞ (infinity norm)
        norm_inf = np.max(g_abs)
        
        # Numerical stability checks
        norm_2_sq = max(0.0, norm_2_sq)
        norm_1 = max(1e-12, norm_1)  # Avoid division by zero
        norm_inf = max(1e-12, norm_inf)  # Avoid division by zero
        
        return norm_2_sq, norm_1, norm_inf
    
    def spectral_objective(self, spectral_coeffs):
        """Objective function that operates on spectral coefficients."""
        # Inverse FFT to get spatial domain function
        f_real = np.real(np.fft.ifft(spectral_coeffs))
        
        # Ensure non-negativity
        f_values = np.maximum(f_real, 0)
        
        # Compute norms
        norm_2_sq, norm_1, norm_inf = self.compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_1 <= 1e-12 or norm_inf <= 1e-12:
            return 1e12
            
        # Minimize negative of C2 to maximize C2
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return -c2 if not np.isnan(c2) and not np.isinf(c2) else 1e12
    
    def optimize_spectral(self):
        """Optimize using spectral domain approach."""
        # Initialize spectral coefficients
        # Start with a structured spectrum that promotes good autoconvolution properties
        init_spectrum = np.zeros(self.n_points, dtype=complex)
        
        # Create a multi-component spectrum with carefully chosen frequencies
        # This creates a rich enough spectrum for good reconstruction while preserving structure
        n_components = 50
        
        for i in range(n_components):
            freq_idx = i + 1  # Avoid DC component
            # Distribute frequencies logarithmically for better coverage
            if freq_idx < self.n_points // 2:
                # Add some frequency components that can create interesting autoconvolution
                magnitude = np.random.exponential(0.1)
                phase = np.random.uniform(0, 2*np.pi)
                init_spectrum[freq_idx] = magnitude * np.exp(1j * phase)
                # Also add symmetric component for real-valued result
                if self.n_points - freq_idx < self.n_points:
                    init_spectrum[self.n_points - freq_idx] = magnitude * np.exp(-1j * phase)
        
        # Add direct current component for baseline
        init_spectrum[0] = np.random.exponential(0.5)
        
        # Optimize using scipy's minimize with L-BFGS-B
        try:
            result = minimize(
                self.spectral_objective,
                init_spectrum,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8},
                bounds=[(None, None) for _ in range(len(init_spectrum))]
            )
            
            if result.success:
                # Convert optimal spectral coeffs back to spatial domain
                optimal_f_real = np.real(np.fft.ifft(result.x))
                return np.maximum(optimal_f_real, 0)
                
        except Exception as e:
            warnings.warn(f"Spectral optimization failed: {str(e)}")
            
        # Fallback to simpler approach
        return self.fallback_method()
    
    def fallback_method(self):
        """Fallback method when spectral optimization fails."""
        # Generate a well-balanced function with careful peak selection
        n_points = self.n_points
        x = np.linspace(-0.25, 0.25, n_points)
        f_values = np.zeros(n_points)
        
        # Create peaks with logarithmic spacing and strategic amplitude variations
        n_peaks = min(20, max(5, n_points // 100))
        
        # Logarithmic distribution of peak positions
        log_min = np.log(0.02)
        log_max = np.log(0.12)
        log_spaced_positions = np.logspace(log_min, log_max, n_peaks // 2 + 1)
        
        peak_positions = []
        peak_amplitudes = []
        peak_widths = []
        
        for i in range(n_peaks):
            side = 1 if i % 2 == 0 else -1
            if i < len(log_spaced_positions):
                pos = side * log_spaced_positions[i // 2] + np.random.uniform(-0.01, 0.01)
            else:
                pos = np.random.uniform(-0.23, 0.23)
                
            pos = np.clip(pos, -0.23, 0.23)
            
            # Ensure minimum spacing
            valid_position = True
            for existing_pos in peak_positions:
                if abs(pos - existing_pos) < 0.02:
                    valid_position = False
                    break
                    
            if valid_position:
                peak_positions.append(pos)
        
        # Generate amplitudes and widths
        for pos in peak_positions:
            center_distance = abs(pos)
            base_amp = np.random.exponential(0.7) * np.exp(-center_distance * 4.0)
            amp = min(1.0, base_amp * np.random.uniform(0.8, 1.2))
            peak_amplitudes.append(amp)
            
            base_sigma = 0.02 + 0.03 * np.exp(-center_distance * 3.0)
            sigma = np.clip(base_sigma * np.random.uniform(0.7, 1.3), 0.005, 0.08)
            peak_widths.append(sigma)
        
        # Add Gaussian peaks
        for i, (pos, amp, sigma) in enumerate(zip(peak_positions, peak_amplitudes, peak_widths)):
            gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
            f_values += gaussian_peak
        
        # Apply smoothing
        if n_points > 100:
            window_size = max(3, min(21, int(n_points / 50)))
            if window_size % 2 == 0:
                window_size += 1
            try:
                f_values = signal.savgol_filter(f_values, window_size, 3)
            except:
                f_values = np.convolve(f_values, np.ones(window_size)/window_size, mode='same')
        
        # Normalize
        max_val = np.max(f_values)
        if max_val > 0:
            f_values = f_values / (max_val * 1.5)
            
        return np.maximum(f_values, 0)

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using spectral optimization."""
    np.random.seed(42)
    
    # Try multiple optimization approaches
    best_c2 = 0.0
    best_f = None
    
    # Try spectral optimization approach
    try:
        optimizer = SpectralPeakOptimizer(n_points=2000)
        f_values = optimizer.optimize_spectral()
        
        # Evaluate result
        if len(f_values) > 0:
            # Compute C2 for this function
            norm_2_sq, norm_1, norm_inf = optimizer.compute_autoconvolution_norms(f_values)
            
            if norm_1 > 1e-12 and norm_inf > 1e-12:
                c2 = norm_2_sq / (norm_1 * norm_inf)
                if c2 > best_c2:
                    best_c2 = c2
                    best_f = f_values.tolist()
    except Exception as e:
        warnings.warn(f"Spectral optimization failed: {str(e)}")
    
    # If spectral approach failed, use fallback
    if best_f is None:
        try:
            optimizer = SpectralPeakOptimizer(n_points=1000)
            f_values = optimizer.fallback_method()
            
            # Evaluate result
            norm_2_sq, norm_1, norm_inf = optimizer.compute_autoconvolution_norms(f_values)
            
            if norm_1 > 1e-12 and norm_inf > 1e-12:
                c2 = norm_2_sq / (norm_1 * norm_inf)
                if c2 > best_c2:
                    best_c2 = c2
                    best_f = f_values.tolist()
        except Exception as e:
            warnings.warn(f"Fallback also failed: {str(e)}")
    
    # If still no success, return simple uniform function
    if best_f is None:
        best_f = [1.0] * 500
    
    return best_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")