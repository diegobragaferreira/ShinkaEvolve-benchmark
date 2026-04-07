# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import differential_evolution
from typing import List, Tuple, Optional
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

class SpectralDomainOptimizer:
    """Optimizes step functions in the spectral domain for superior C2 values."""
    
    def __init__(self):
        self.max_time_seconds = 85
        self.min_steps = 100
        self.max_steps = 10000
        
    def compute_autoconvolution_norms(self, f_values: List[float]) -> Tuple[float, float, float]:
        """
        Compute autoconvolution norms using FFT for efficiency and accuracy.
        This is the core computational engine that matches the evaluator's method.
        """
        if not f_values or len(f_values) == 0:
            return 0.0, 0.0, 0.0
            
        # Convert to numpy array
        f = np.array(f_values, dtype=np.float64)
        n = len(f)
        
        # Compute autoconvolution using FFT: conv(f,f) = ifft(fft(f)^2)
        fft_f = fft(f)
        fft_g = fft_f * fft_f.conj()  # Convolution in frequency domain
        g = ifft(fft_g).real  # Convert back to real domain
        
        # Extract central portion (valid autoconvolution)
        half_len = n - 1
        if len(g) >= half_len:
            g = g[half_len:]  # Take right half
        else:
            return 0.0, 0.0, 0.0

        # Compute norms with proper numerical handling
        g_squared = g * g
        norm_2_sq = np.sum(g_squared)  # L2 norm squared
        
        norm_1 = np.sum(np.abs(g))  # L1 norm
        norm_inf = np.max(np.abs(g))  # L-infinity norm
        
        # Prevent numerical issues
        norm_1 = max(1e-15, norm_1)
        norm_inf = max(1e-15, norm_inf)
        
        return norm_2_sq, norm_1, norm_inf

    def compute_c2(self, f_values: List[float]) -> float:
        """Compute C2 value with robust error handling."""
        try:
            norm_2_sq, norm_1, norm_inf = self.compute_autoconvolution_norms(f_values)
            
            # Avoid division by zero
            if norm_1 <= 1e-12 or norm_inf <= 1e-12:
                return 0.0
                
            c2 = norm_2_sq / (norm_1 * norm_inf)
            return float(c2) if not np.isnan(c2) and not np.isinf(c2) else 0.0
        except Exception:
            return 0.0

    def create_spectral_peak_function(self, n_steps: int) -> List[float]:
        """
        Create function by designing its frequency spectrum for optimal autoconvolution.
        This is the key innovation - working directly in spectral domain.
        """
        if n_steps < self.min_steps:
            n_steps = self.min_steps
            
        # Generate frequency domain representation
        freqs = fftfreq(n_steps, 1.0/n_steps)
        
        # Create magnitude spectrum that promotes favorable autoconvolution
        magnitudes = np.zeros(n_steps)
        
        # Add central concentration to encourage flat autoconvolution
        # This is based on the principle that concentrated power spectra lead to 
        # smoother autoconvolutions
        central_fraction = 0.4  # Concentrate 40% of energy in center
        center_indices = int(n_steps * central_fraction)
        
        # Add multiple frequency components with strategic placement
        n_components = min(15, n_steps // 10)
        
        # Create log-spaced frequency components for broad coverage
        if n_components > 0:
            log_indices = np.logspace(np.log10(1), np.log10(n_steps//4), n_components, base=10, dtype=int)
            log_indices = log_indices[log_indices < n_steps//2]
            
            for i, freq_idx in enumerate(log_indices):
                if freq_idx < n_steps//2:
                    # Apply decreasing strength with log distribution
                    strength = 1.0 / (1.0 + i * 0.3)
                    magnitudes[freq_idx] = strength
                    if freq_idx > 0:
                        magnitudes[-freq_idx] = strength
        
        # Add strong DC component for stability
        magnitudes[0] = 1.0
        
        # Add some medium frequency components for texture
        mid_freqs = [int(n_steps * 0.1), int(n_steps * 0.2), int(n_steps * 0.3)]
        for freq_idx in mid_freqs:
            if freq_idx < n_steps//2:
                # Medium strength components
                strength = 0.5 / (1.0 + freq_idx * 0.01)
                magnitudes[freq_idx] = strength
                if freq_idx > 0:
                    magnitudes[-freq_idx] = strength

        # Add random phases to avoid local minima
        phases = np.random.uniform(0, 2*np.pi, n_steps)
        
        # Create complex spectrum
        complex_spectrum = magnitudes * np.exp(1j * phases)
        
        # Ensure conjugate symmetry for real-valued output
        complex_spectrum = self._ensure_conjugate_symmetry(complex_spectrum)
        
        # Convert back to time domain using inverse FFT
        f_real = np.real(ifft(complex_spectrum))
        
        # Ensure non-negativity and normalize
        f_real = np.maximum(f_real, 0.0)
        
        # Normalize appropriately
        max_val = np.max(f_real)
        if max_val > 0:
            f_real = f_real / (max_val * 1.5)
        
        # Apply gentle smoothing to reduce numerical artifacts
        if n_steps > 50:
            try:
                # Simple moving average smoothing
                window_size = min(21, n_steps // 10)
                if window_size % 2 == 0:
                    window_size += 1
                kernel = np.ones(window_size) / window_size
                f_real = np.convolve(f_real, kernel, mode='same')
            except:
                pass
        
        # Clamp to ensure non-negativity
        f_real = np.maximum(f_real, 0.0)
        
        return f_real.tolist()

    def _ensure_conjugate_symmetry(self, spectrum: np.ndarray) -> np.ndarray:
        """Ensure conjugate symmetry for real-valued time domain signals."""
        n = len(spectrum)
        sym_spectrum = spectrum.copy()
        
        # For real signals, spectrum[k] = spectrum[N-k]* (conjugate symmetry)
        if n % 2 == 0:
            # Even length - Nyquist component
            sym_spectrum[n//2] = np.real(sym_spectrum[n//2])
        else:
            # Odd length - no Nyquist
            pass
        
        # Apply conjugate symmetry for other components
        for k in range(1, n//2):
            conj_k = n - k
            if conj_k < n:
                avg_val = (sym_spectrum[k] + sym_spectrum[conj_k].conj()) / 2.0
                sym_spectrum[k] = avg_val
                sym_spectrum[conj_k] = avg_val.conj()
        
        return sym_spectrum

    def create_adaptive_peak_function(self, n_steps: int) -> List[float]:
        """Create peak-based function with adaptive positioning and parameters."""
        # Create base function with controlled peak distribution
        x = np.linspace(-0.25, 0.25, n_steps)
        f = np.zeros(n_steps)
        
        # Use adaptive log-spacing for peak positions
        n_peaks = max(4, min(16, n_steps // 100))
        
        # Create log-spaced peak positions for broad coverage
        log_positions = np.logspace(np.log10(0.02), np.log10(0.48), n_peaks, base=np.e)
        
        peak_positions = []
        peak_widths = []
        peak_heights = []
        
        # Generate peaks with better statistical distribution
        for i, log_pos in enumerate(log_positions):
            # Map to actual coordinate
            pos = -0.25 + 0.03 + log_pos * (0.5 - 2*0.03)  # Leave margin
            pos = np.clip(pos, -0.25 + 0.03, 0.25 - 0.03)
            peak_positions.append(pos)
            
            # Generate peak parameters
            width = np.random.uniform(0.008, 0.025)
            peak_widths.append(width)
            
            # Height inversely proportional to width for better control
            height = np.random.uniform(0.7, 2.0)
            peak_heights.append(height)
        
        # Create Gaussian peaks
        for pos, width, height in zip(peak_positions, peak_widths, peak_heights):
            gaussian = height * np.exp(-0.5 * ((x - pos) / width) ** 2)
            f += gaussian
        
        # Apply smoothing for numerical stability
        if n_steps > 50:
            try:
                from scipy.ndimage import gaussian_filter1d
                f = gaussian_filter1d(f, sigma=0.5)
            except:
                pass
        
        # Ensure non-negativity
        f = np.maximum(f, 0.0)
        
        # Normalize appropriately
        max_val = np.max(f)
        if max_val > 0:
            f = f / (max_val * 1.2)
        
        return f.tolist()

    def construct_optimized_function(self, max_time_seconds: int = 85) -> List[float]:
        """
        Main function construction routine with improved architecture.
        Uses multiple strategies and parallel evaluation.
        """
        start_time = time.time()
        best_c2 = 0.0
        best_function = []
        
        # Determine function size based on time budget
        n_steps = min(self.max_steps, max(self.min_steps, 1000 + int(np.random.randint(0, 500) * 5)))
        
        # Strategy 1: Spectral domain optimization
        try:
            s1_func = self.create_spectral_peak_function(n_steps)
            s1_c2 = self.compute_c2(s1_func)
            
            if s1_c2 > best_c2:
                best_c2 = s1_c2
                best_function = s1_func.copy()
        except Exception as e:
            pass
            
        # Strategy 2: Adaptive peak function
        try:
            s2_func = self.create_adaptive_peak_function(n_steps)
            s2_c2 = self.compute_c2(s2_func)
            
            if s2_c2 > best_c2:
                best_c2 = s2_c2
                best_function = s2_func.copy()
        except Exception as e:
            pass
            
        # Strategy 3: Multi-strategy with parallel evaluation
        if time.time() - start_time < max_time_seconds - 10:
            try:
                strategies = []
                
                def evaluate_strategy(strategy_func, strategy_name):
                    try:
                        c2 = self.compute_c2(strategy_func)
                        return (c2, strategy_func, strategy_name)
                    except Exception:
                        return (0.0, [], strategy_name)
                
                # Generate multiple candidate functions in parallel
                candidates = [
                    ("adaptive", self.create_adaptive_peak_function(n_steps)),
                    ("spectral", self.create_spectral_peak_function(n_steps)),
                    ("simple_peak", self._create_simple_peak_function(n_steps)),
                ]
                
                # Evaluate in parallel
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [
                        executor.submit(evaluate_strategy, func, name) 
                        for name, func in candidates
                    ]
                    
                    results = [future.result() for future in as_completed(futures)]
                    
                # Find best among parallel results
                for c2, func, name in results:
                    if c2 > best_c2:
                        best_c2 = c2
                        best_function = func.copy()
                        
            except Exception:
                pass
        
        # Final refinement if we have a candidate
        if best_function and time.time() - start_time < max_time_seconds - 5:
            try:
                # Try local refinement
                refined_func = self._local_refinement(best_function, n_steps)
                refined_c2 = self.compute_c2(refined_func)
                
                if refined_c2 > best_c2:
                    best_c2 = refined_c2
                    best_function = refined_func
            except Exception:
                pass
        
        # Fallback to simple function if nothing worked well
        if not best_function:
            best_function = [0.5] * n_steps
            
        return best_function

    def _create_simple_peak_function(self, n_steps: int) -> List[float]:
        """Create a simple peak-based function for fallback."""
        x = np.linspace(-0.25, 0.25, n_steps)
        f = np.zeros(n_steps)
        
        # Single dominant peak
        peak_height = 1.0
        peak_width = 0.02
        peak_center = 0.0
        f += peak_height * np.exp(-0.5 * ((x - peak_center) / peak_width) ** 2)
        
        # Normalize
        max_val = np.max(f)
        if max_val > 0:
            f = f / (max_val * 1.2)
            
        return f.tolist()

    def _local_refinement(self, initial_function: List[float], n_steps: int) -> List[float]:
        """Apply local refinement to improve C2."""
        try:
            # Simple gradient-free refinement
            f_array = np.array(initial_function)
            current_c2 = self.compute_c2(f_array.tolist())
            best_f = f_array.copy()
            best_c2 = current_c2
            
            # Try small perturbations
            for _ in range(50):  # Limited iterations for speed
                # Create perturbed version
                perturbed = f_array.copy()
                idx = np.random.randint(len(perturbed))
                # Small random perturbation
                delta = np.random.normal(0, 0.05)
                perturbed[idx] = max(0, perturbed[idx] + delta)
                
                # Evaluate
                new_c2 = self.compute_c2(perturbed.tolist())
                if new_c2 > best_c2:
                    best_c2 = new_c2
                    best_f = perturbed.copy()
                    
            return best_f.tolist()
            
        except Exception:
            return initial_function

class EvolutionaryOptimizer:
    """Legacy evolutionary approach for comparison and fallback."""
    
    def __init__(self):
        self.c2_computer = SpectralDomainOptimizer()
        
    def create_structured_gaussian_individual(self, n_steps: int) -> List[float]:
        """Create structured individual using Gaussian peaks."""
        # Create step function with Gaussian peaks
        x = np.linspace(-0.25, 0.25, n_steps)
        f_vals = np.zeros(n_steps)
        
        # Use logarithmic spacing for peak positions
        n_peaks = max(3, min(15, n_steps // 100))
        
        # Distribute peaks with log spacing
        log_min = np.log(0.05)
        log_max = np.log(0.45)
        log_positions = np.logspace(log_min, log_max, n_peaks, base=np.e)
        
        # Map to actual positions
        total_range = 0.5
        offset = 0.05
        
        peak_positions = []
        peak_widths = []
        peak_heights = []
        
        for i in range(n_peaks):
            rel_pos = log_positions[i] if i < len(log_positions) else 0.5
            pos = -0.25 + offset + rel_pos * (total_range - 2*offset)
            pos = np.clip(pos, -0.25 + offset, 0.25 - offset)
            peak_positions.append(pos)
            
            # Add small random perturbation
            peak_positions[i] += np.random.uniform(-0.015, 0.015)
            peak_positions[i] = np.clip(peak_positions[i], -0.25 + 0.05, 0.25 - 0.05)
            
            # Generate peak parameters
            width = np.random.uniform(0.005, 0.025)
            peak_widths.append(width)
            
            height = np.random.uniform(0.8, 2.0)
            peak_heights.append(height)
        
        # Create Gaussian curves
        for center, width, height in zip(peak_positions, peak_widths, peak_heights):
            gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
            f_vals += gaussian
        
        # Apply smoothing
        if n_steps > 50:
            from scipy.ndimage import gaussian_filter1d
            f_vals = gaussian_filter1d(f_vals, sigma=0.8)
        
        # Ensure non-negativity
        f_vals = np.maximum(f_vals, 0)
        
        # Normalize
        if np.max(f_vals) > 0:
            f_vals = f_vals / np.max(f_vals) * 1.5
            
        return f_vals.tolist()

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses the new spectral domain optimization approach.
    """
    # For compatibility with existing interface
    optimizer = SpectralDomainOptimizer()
    return optimizer.construct_optimized_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")