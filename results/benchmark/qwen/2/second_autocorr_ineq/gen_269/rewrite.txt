# EVOLVE-BLOCK-START

import numpy as np
import numba
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List
import time
from joblib import Parallel, delayed
import warnings
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# JIT compile the core computation functions for speed
@numba.jit(nopython=True)
def compute_autoconvolution_fast(f_vals):
    """Fast autoconvolution computation using Numba"""
    n = len(f_vals)
    g = np.zeros(2 * n - 1)

    # Manual convolution for speed
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]

    return g

@numba.jit(nopython=True)
def compute_norms_piecewise(g_vals):
    """Compute norms using piecewise linear integration matching evaluator's method"""
    n = len(g_vals)

    if n <= 1:
        return 0.0, 0.0, 0.0

    # Compute L2 norm squared using trapezoidal-like integration
    # Formula: (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
    norm_2_sq = 0.0
    dx = 0.5 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.5

    for i in range(n - 1):
        y1 = g_vals[i]
        y2 = g_vals[i + 1]
        norm_2_sq += (dx / 3.0) * (y1 * y1 + y1 * y2 + y2 * y2)

    # Compute L1 norm (sum of absolute values)
    norm_1 = 0.0
    for i in range(n):
        norm_1 += abs(g_vals[i])

    # Compute L-infinity norm (maximum absolute value)
    norm_inf = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > norm_inf:
            norm_inf = abs_val

    return norm_2_sq, norm_1, norm_inf

def compute_autoconvolution_norms(f: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation using efficient piecewise integration.
    Returns (||g||₂², ||g||₁, ||g||∞)
    """
    # Convert to numpy array
    f_arr = np.array(f, dtype=np.float64)

    # Compute autoconvolution
    g = compute_autoconvolution_fast(f_arr)

    # Compute norms using piecewise integration
    norm_2_sq, norm_1, norm_inf = compute_norms_piecewise(g)

    return norm_2_sq, norm_1, norm_inf

def compute_c2(f: List[float]) -> float:
    """Compute C2 value for given function"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f)

    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0

    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

class PeakGenerator:
    """Generates strategic peak distributions for optimal autoconvolution properties."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_multi_scale_peaks(self, n_steps: int, peak_count: int = None) -> List[tuple]:
        """
        Generate multi-scale peaks with strategic placement and characteristics.
        """
        if peak_count is None:
            peak_count = max(10, min(50, n_steps // 100))

        x = np.linspace(-0.25, 0.25, n_steps)
        peaks = []

        # Enhanced approach: Generate peaks with better strategic positioning
        # Using a hybrid approach that combines different spacing strategies

        # Approach 1: Golden ratio based distribution for even spread
        golden_ratio = (1 + np.sqrt(5)) / 2
        golden_positions = []
        for i in range(peak_count):
            # Distribute using golden ratio for better equidistribution
            pos = -0.25 + (i * golden_ratio) % 1.0 * 0.5
            golden_positions.append(pos)

        # Approach 2: Strategic clustering around key regions
        cluster_positions = []
        # Add clusters around center for better convolution
        for i in range(3):
            cluster_center = (i - 1) * 0.1  # -0.1, 0, 0.1
            for j in range(2):
                offset = (j - 0.5) * 0.05  # Small offsets
                cluster_positions.append(cluster_center + offset)

        # Approach 3: Uniform distribution with strategic gaps
        uniform_positions = np.linspace(-0.2, 0.2, max(5, peak_count // 2), endpoint=True)

        # Combine and select best positions
        all_positions = np.array(golden_positions + cluster_positions + uniform_positions.tolist())

        # Select unique positions ensuring minimum separation
        selected_positions = []
        for pos in all_positions:
            # Check minimum separation
            valid = True
            for existing_pos in selected_positions:
                if abs(pos - existing_pos) < 0.01:
                    valid = False
                    break
            if valid and len(selected_positions) < peak_count:
                selected_positions.append(pos)

        # If we don't have enough peaks, fill with uniformly distributed ones
        if len(selected_positions) < peak_count:
            additional = peak_count - len(selected_positions)
            extra_positions = np.linspace(-0.24, 0.24, additional + 2)[1:-1]  # Excluding endpoints
            selected_positions.extend(extra_positions[:additional])

        selected_positions = selected_positions[:peak_count]

        # Generate corresponding heights and widths
        # Heights: prefer moderate heights to avoid overly sharp autoconvolution
        heights = np.random.uniform(1.2, 2.2, len(selected_positions))

        # Widths: adjust based on position and height to optimize autoconvolution
        widths = []
        for i, (pos, height) in enumerate(zip(selected_positions, heights)):
            # Position-dependent width: wider near edges, narrower in center
            edge_factor = 1.0 - 0.5 * abs(pos) / 0.25  # 0.5 at edges, 1.0 at center
            # Height-dependent width: narrower for taller peaks
            height_factor = 1.0 / (1.0 + 0.3 * (height - 1.2))
            base_width = 0.01 + 0.01 * edge_factor * height_factor
            # Add some randomness
            width = base_width * np.random.uniform(0.8, 1.2)
            widths.append(width)

        # Convert to list of tuples
        peaks = list(zip(selected_positions, heights, widths))

        # Enhance peak qualities based on position
        enhanced_peaks = self._enhance_peaks(peaks)

        return enhanced_peaks

    def _logarithmic_distribution(self, start: float, end: float, count: int) -> np.ndarray:
        """Generate logarithmic distribution of points."""
        if count <= 1:
            return np.array([(start + end) / 2])

        # Create logarithmic spacing
        log_start = np.log(abs(start) + 1e-10)
        log_end = np.log(abs(end) + 1e-10)
        log_points = np.linspace(log_start, log_end, count)
        points = np.sign(end) * np.exp(log_points)

        # Clip to range
        points = np.clip(points, start, end)
        return points

    def _filter_peaks(self, positions: np.ndarray, heights: np.ndarray, widths: np.ndarray) -> List[tuple]:
        """Filter peaks by minimum spatial separation and quality."""
        filtered_peaks = []
        min_gap = 0.01

        # Sort peaks by position for consistent processing
        sorted_indices = np.argsort(positions)
        sorted_positions = positions[sorted_indices]
        sorted_heights = heights[sorted_indices]
        sorted_widths = widths[sorted_indices]

        for i, (pos, height, width) in enumerate(zip(sorted_positions, sorted_heights, sorted_widths)):
            # Skip if peak is too close to existing peaks
            valid = True
            for existing_pos, _, _ in filtered_peaks:
                if abs(pos - existing_pos) < min_gap:
                    valid = False
                    break

            # Skip peaks that would create poor autoconvolution profiles
            if valid:
                # Quality check: avoid extremely narrow peaks that might cause numerical issues
                if width < 0.002 or width > 0.1:
                    continue
                filtered_peaks.append((pos, height, width))

        return filtered_peaks

    def _enhance_peaks(self, peaks: List[tuple]) -> List[tuple]:
        """Enhance peak characteristics based on position and other factors."""
        enhanced = []
        for pos, height, width in peaks:
            # Reduce height for peaks near boundaries
            if abs(pos) > 0.15:
                height *= 0.8

            # Adjust width based on height to maintain good autoconvolution properties
            if height > 2.0:
                width *= 0.8
            elif height < 1.2:
                width *= 1.2

            enhanced.append((pos, height, width))
        return enhanced

class FunctionBuilder:
    """Constructs functions from peak specifications."""

    def __init__(self):
        pass

    def build_from_peaks(self, peaks: List[tuple], n_steps: int) -> np.ndarray:
        """Build function from peak specifications."""
        x = np.linspace(-0.25, 0.25, n_steps)
        f_values = np.zeros(n_steps)

        # Apply all peaks
        for pos, height, width in peaks:
            gaussian = height * np.exp(-0.5 * ((x - pos) / width)**2)
            f_values += gaussian

        # Add supplementary structure
        self._add_supplementary_structure(f_values, x)

        # Ensure non-negativity and normalize
        f_values = np.maximum(f_values, 0)
        if np.max(f_values) > 0:
            f_values = f_values / np.max(f_values) * 1.8

        return f_values

    def _add_supplementary_structure(self, f_values: np.ndarray, x: np.ndarray):
        """Add supplementary structure for better autoconvolution properties."""
        n_steps = len(f_values)
        for i in range(0, n_steps, max(1, n_steps // 40)):
            if np.random.random() > 0.8:
                bump_center = x[i]
                bump_height = np.random.uniform(0.05, 0.3)
                bump_width = np.random.uniform(0.005, 0.015)
                bump = bump_height * np.exp(-0.5 * ((x - bump_center) / bump_width)**2)
                f_values += bump

class Optimizer:
    """Performs local optimization on functions."""

    def __init__(self):
        pass

    def optimize(self, func_vals: List[float]) -> List[float]:
        """Perform local optimization using targeted approaches."""
        current_func = np.array(func_vals)
        best_c2 = self._compute_c2(current_func)
        best_func = current_func.copy()

        # Multi-stage optimization focusing on different aspects
        # Stage 1: Peak-focused optimization
        optimized_func = self._optimize_peaks(current_func)
        stage_c2 = self._compute_c2(optimized_func)
        if stage_c2 > best_c2:
            best_c2 = stage_c2
            best_func = optimized_func.copy()

        # Stage 2: Fine-tuning with neighborhood search
        for _ in range(2):
            for _ in range(50):
                test_func = best_func.copy()
                # Focus on significant feature points
                indices = np.where(test_func > np.percentile(test_func, 60))[0]
                if len(indices) > 0:
                    idx = np.random.choice(indices)
                    adjustment = np.random.normal(0, 0.02 * test_func[idx])
                    test_func[idx] = max(0, test_func[idx] + adjustment)
                else:
                    idx = np.random.randint(0, len(test_func))
                    adjustment = np.random.normal(0, 0.02)
                    test_func[idx] = max(0, test_func[idx] + adjustment)

                test_c2 = self._compute_c2(test_func)
                if test_c2 > best_c2:
                    best_c2 = test_c2
                    best_func = test_func.copy()

        return best_func.tolist()

    def _optimize_peaks(self, func_vals: np.ndarray) -> np.ndarray:
        """Targeted optimization focused on peak characteristics."""
        # Identify peak locations and estimate their properties
        peaks = self._find_peaks(func_vals)

        # For demonstration, we'll just do a focused search around peaks
        # In practice, this would be more sophisticated
        test_func = func_vals.copy()
        return test_func

    def _find_peaks(self, func_vals: np.ndarray) -> List[int]:
        """Find indices where peaks occur."""
        peaks = []
        for i in range(1, len(func_vals)-1):
            if func_vals[i] > func_vals[i-1] and func_vals[i] > func_vals[i+1]:
                peaks.append(i)
        return peaks

    def _compute_c2(self, func_vals: np.ndarray) -> float:
        """Compute C₂ value with numerical stability."""
        f = np.array(func_vals)

        # Autoconvolution using convolution
        g = np.convolve(f, f, mode='full')
        g = g[len(g)//2:]

        # Adjust for correct length
        if len(g) > len(f):
            g = g[:len(f)]

        # Compute norms using proper integration method matching evaluator
        dx = 0.5 / (len(f) - 1) if len(f) > 1 else 0.5

        # L2 norm squared using trapezoidal-like integration
        norm_2_sq = 0.0
        for i in range(len(g)-1):
            y1 = g[i]
            y2 = g[i+1]
            norm_2_sq += (dx / 3.0) * (y1 * y1 + y1 * y2 + y2 * y2)

        # L1 norm (sum of absolute values)
        norm_1 = 0.0
        for i in range(len(g)):
            norm_1 += abs(g[i])

        # L-infinity norm (maximum absolute value)
        norm_inf = 0.0
        for i in range(len(g)):
            abs_val = abs(g[i])
            if abs_val > norm_inf:
                norm_inf = abs_val

        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0

        return norm_2_sq / (norm_1 * norm_inf)

def spectral_guided_initialization(n_steps: int) -> List[float]:
    """
    Create initial function using spectral guidance:
    - Construct function with low-frequency dominance
    - Add structured variations to avoid trivial solutions
    - Ensure properties that support high C2 values
    """
    # Create base function with spectral characteristics known to work well
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Use a combination of sinusoidal components and Gaussian features
    # Low-frequency dominant spectrum tends to produce better C2 values
    f_vals = np.zeros(n_steps)
    
    # Add a main Gaussian envelope (central peak)
    main_peak = 2.0 * np.exp(-0.5 * (x / 0.08)**2)
    f_vals += main_peak
    
    # Add secondary components with different frequencies and phases
    # These are carefully chosen to create constructive interference in autoconvolution
    frequencies = [2.0, 3.0, 5.0]  # Low frequencies that promote good autoconvolution
    phases = [0.0, np.pi/4, np.pi/2]
    
    for freq, phase in zip(frequencies, phases):
        # Add sinusoidal modulations
        modulation = 0.5 * np.sin(freq * np.pi * x + phase)
        f_vals += modulation * np.exp(-0.5 * (x / 0.1)**2)  # Localize the modulation
    
    # Add some step-like features to provide variety
    if n_steps > 100:
        n_regions = min(8, n_steps // 50)
        for i in range(n_regions):
            start_idx = int(i * n_steps / n_regions)
            end_idx = int((i + 1) * n_steps / n_regions)
            if i % 2 == 0:
                f_vals[start_idx:end_idx] += 0.3
    
    # Ensure non-negativity and normalize
    f_vals = np.maximum(f_vals, 0)
    
    # Apply mild smoothing to ensure differentiability
    if n_steps > 20:
        kernel = np.ones(7) / 7
        f_vals = np.convolve(f_vals, kernel, mode='same')
    
    # Normalize to reasonable scale
    if np.max(f_vals) > 0:
        f_vals = f_vals / np.max(f_vals) * 1.5
    
    return f_vals.tolist()

def peak_based_initialization(n_steps: int) -> List[float]:
    """
    Create initial function using peak-based generation
    """
    peak_generator = PeakGenerator(seed=42)
    function_builder = FunctionBuilder()
    
    # Generate peaks with strategic positioning
    peaks = peak_generator.generate_multi_scale_peaks(n_steps)
    
    # Build function from peaks
    f_values = function_builder.build_from_peaks(peaks, n_steps)
    
    # Apply smoothing
    f_values = _smooth_function(f_values, n_steps)
    
    return f_values.tolist()

def _smooth_function(f_values: np.ndarray, n_steps: int) -> np.ndarray:
    """Apply smoothing to reduce sharp transitions."""
    # Adaptive window size based on function characteristics
    # If function has many peaks, use smaller window to preserve detail
    # If function is relatively smooth, use larger window for more smoothing
    peak_count = len(np.where(np.diff(np.sign(np.diff(f_values))) != 0)[0])
    base_window = max(3, min(51, n_steps // 100))
    window_size = min(51, max(3, base_window + peak_count // 10))

    if window_size % 2 == 0:
        window_size += 1
    if window_size > 1:
        # Use Gaussian window for smoother results
        sigma = window_size / 6.0
        window = np.exp(-0.5 * np.square(np.arange(window_size) - window_size // 2) / sigma**2)
        window = window / np.sum(window)
        f_values = np.convolve(f_values, window, mode='same')
    return f_values

def fourier_domain_refinement(initial_f: List[float], max_iter: int = 20) -> List[float]:
    """
    Refine function using Fourier domain techniques:
    - Transform to frequency domain
    - Apply spectral modifications that enhance C2
    - Transform back to time domain
    """
    f_current = np.array(initial_f, dtype=np.float64)
    best_c2 = compute_c2(f_current.tolist())
    best_f = f_current.copy()
    
    # Sample the frequency domain representation
    # Use FFT to analyze the spectral content
    f_fft = fft(f_current)
    f_freq = fftfreq(len(f_current), 0.5 / (len(f_current) - 1))
    
    # Create a mask for spectral enhancement
    # Target suppression of very high frequencies that might cause instability
    # and enhancement of medium frequencies that promote good autoconvolution
    
    # Convert to magnitude spectrum
    mag_spectrum = np.abs(f_fft)
    
    # Apply spectral shaping - emphasize medium frequencies
    # This helps create smoother autoconvolutions with better norm ratios
    spectral_weights = np.ones_like(mag_spectrum)
    
    # Emphasize middle frequencies (avoid both DC and very high frequencies)
    middle_freq_indices = np.where((np.abs(f_freq) > 0.1) & (np.abs(f_freq) < 10))[0]
    spectral_weights[middle_freq_indices] *= 1.2  # Boost middle frequencies
    
    # Reduce very high frequencies
    high_freq_indices = np.where(np.abs(f_freq) > 15)[0]
    spectral_weights[high_freq_indices] *= 0.7
    
    # Apply weights to spectral components
    modified_spectrum = f_fft * spectral_weights
    
    # Reconstruct signal
    f_reconstructed = np.real(ifft(modified_spectrum))
    
    # Ensure non-negativity
    f_reconstructed = np.maximum(f_reconstructed, 0)
    
    # Evaluate and accept if better
    new_c2 = compute_c2(f_reconstructed.tolist())
    if new_c2 > best_c2:
        best_c2 = new_c2
        best_f = f_reconstructed.copy()
    
    # Additional gradient-based refinement in time domain if needed
    for iteration in range(max_iter):
        # Simple gradient-like update based on sensitivity analysis
        f_new = best_f.copy()
        
        # Perturb some key points based on spectral analysis
        # Focus on areas where spectral energy is concentrated
        energy_distribution = np.abs(f_reconstructed)**2
        
        # Select top 10% of energy locations for adjustment
        top_energy_indices = np.argsort(energy_distribution)[-max(5, len(f_new)//20):]
        
        # Small perturbations around high-energy regions
        for idx in top_energy_indices:
            # Perturb with adaptive magnitude based on local energy
            perturbation_magnitude = 0.03 * f_new[idx] if f_new[idx] > 0 else 0.05
            perturbation = np.random.normal(0, perturbation_magnitude)
            f_new[idx] = max(0, f_new[idx] + perturbation)
        
        # Evaluate new function
        new_c2 = compute_c2(f_new.tolist())
        
        # Accept improvement
        if new_c2 > best_c2:
            best_c2 = new_c2
            best_f = f_new.copy()
    
    return best_f.tolist()

def adaptive_spectral_optimization(initial_f: List[float], max_evals: int = 150) -> List[float]:
    """
    Use a hybrid approach combining spectral analysis with direct optimization
    """
    try:
        # Start with spectral-guided refinement
        refined_f = fourier_domain_refinement(initial_f, max_iter=10)
        current_c2 = compute_c2(refined_f)
        
        # If still getting better, apply more aggressive spectral tuning
        if current_c2 > 0.7:  # Threshold for when to apply advanced spectral methods
            # Create a more sophisticated spectral modification
            f_refined = np.array(refined_f, dtype=np.float64)
            
            # Get frequency domain
            f_fft = fft(f_refined)
            freq = fftfreq(len(f_refined), 0.5 / (len(f_refined) - 1))
            
            # Apply adaptive spectral shaping
            # We want to suppress high-frequency noise while preserving useful features
            # Use a smooth filter that attenuates frequencies above a certain threshold
            threshold_freq = 8.0
            # Create a window function that tapers high frequencies
            window = np.exp(-0.5 * (freq / threshold_freq)**2)
            
            # Apply tapering to the spectrum
            modified_fft = f_fft * window
            
            # Reconstruct
            reconstructed = np.real(ifft(modified_fft))
            reconstructed = np.maximum(reconstructed, 0)
            
            # Evaluate
            new_c2 = compute_c2(reconstructed.tolist())
            if new_c2 > current_c2:
                refined_f = reconstructed.tolist()
                current_c2 = new_c2
        
        # Final local refinement if beneficial
        if current_c2 > 0.7:
            # Use a more targeted local search on the refined function
            refined_local = local_search_refinement(refined_f, max_iter=15)
            local_c2 = compute_c2(refined_local)
            if local_c2 > current_c2:
                refined_f = refined_local
        
        return refined_f
        
    except Exception as e:
        # Fallback to standard approach if anything fails
        return initial_f

def local_search_refinement(initial_f: List[float], max_iter: int = 30) -> List[float]:
    """
    Apply local search to improve the function with enhanced targeting
    """
    f_current = np.array(initial_f, dtype=np.float64)
    best_c2 = compute_c2(f_current.tolist())
    best_f = f_current.copy()

    # Enhanced local search that focuses on key characteristics
    for iteration in range(max_iter):
        # Create neighbor by making small changes
        f_new = f_current.copy()

        # Improved selection strategy: target the top 15% of values
        # This focuses on the most influential parts of the function
        sorted_indices = np.argsort(f_current)[::-1]  # descending order
        indices_to_modify = sorted_indices[:max(5, len(f_current) // 15)]

        # Adaptive perturbation based on function characteristics
        for idx in indices_to_modify:
            # Use larger perturbations for higher values to explore more
            if f_new[idx] > np.percentile(f_current, 70):
                # Large perturbation for significant peaks
                perturbation = np.random.normal(0, 0.1 * f_new[idx])
            elif f_new[idx] > np.percentile(f_current, 30):
                # Medium perturbation for mid-range values  
                perturbation = np.random.normal(0, 0.05 * f_new[idx])
            else:
                # Small perturbation for background levels
                perturbation = np.random.normal(0, 0.02 * f_new[idx]) if f_new[idx] > 0 else np.random.normal(0, 0.05)

            f_new[idx] = max(0, f_new[idx] + perturbation)

        # Evaluate new function
        new_c2 = compute_c2(f_new.tolist())

        # Accept improvement
        if new_c2 > best_c2:
            best_c2 = new_c2
            best_f = f_new.copy()

        f_current = f_new

    return best_f.tolist()

def evaluate_candidate(individual: List[float]) -> float:
    """Evaluate a single candidate function"""
    return compute_c2(individual)

def construct_function() -> List[float]:
    """
    Construct step function with high C2 value using hybrid peak-spectral evolution approach
    """
    start_time = time.time()

    # Set up parameters
    max_time_seconds = 85

    # Strategy 1: Generate a few highly optimized structured functions using hybrid initialization
    best_c2 = 0.0
    best_function = []

    # Try different configurations with hybrid initialization
    configs = []
    
    # Different resolutions and structures
    n_steps_options = [1000, 2000, 3000, 4000, 5000]
    for n_steps in n_steps_options:
        # Create several variants for diversity - mix peak-based and spectral-based
        for variant in range(2):
            configs.append({'steps': n_steps, 'variant': variant, 'method': 'peak' if variant == 0 else 'spectral'})

    # Evaluate candidates with hybrid initialization
    all_candidates = []
    
    for config in configs:
        n_steps = config['steps']
        method = config['method']
        
        if method == 'peak':
            # Use peak-based initialization
            f_init = peak_based_initialization(n_steps)
        else:
            # Use spectral-guided initialization
            f_init = spectral_guided_initialization(n_steps)
        
        # Add slight randomness to break symmetries
        f_init = [val * (0.9 + random.random() * 0.2) for val in f_init]
        
        all_candidates.append(f_init)
        
        # Early exit if time is running out
        if time.time() - start_time > max_time_seconds - 10:
            break

    # Evaluate candidates (with parallel processing for efficiency)
    if all_candidates:
        try:
            # Batch evaluation to reduce overhead 
            batch_size = min(10, len(all_candidates))
            all_fitness_scores = []
            
            for i in range(0, len(all_candidates), batch_size):
                batch = all_candidates[i:i+batch_size]
                batch_scores = Parallel(n_jobs=min(4, len(batch)))(
                    delayed(evaluate_candidate)(candidate) for candidate in batch
                )
                all_fitness_scores.extend(batch_scores)
            
            # Find best candidate
            best_idx = np.argmax(all_fitness_scores)
            best_c2 = all_fitness_scores[best_idx]
            best_function = all_candidates[best_idx].copy()
            
        except Exception:
            # Fallback to sequential evaluation if parallel fails
            best_c2 = 0.0
            best_function = []
            for i, candidate in enumerate(all_candidates):
                if time.time() - start_time > max_time_seconds - 10:
                    break
                score = evaluate_candidate(candidate)
                if score > best_c2:
                    best_c2 = score
                    best_function = candidate.copy()

    # Final refinement using hybrid optimization if we have a candidate
    if best_function and time.time() - start_time < max_time_seconds - 5:
        # Apply hybrid spectral-guided refinement
        refined_hybrid = adaptive_spectral_optimization(best_function, max_evals=150)
        hybrid_c2 = compute_c2(refined_hybrid)
        
        if hybrid_c2 > best_c2:
            best_c2 = hybrid_c2
            best_function = refined_hybrid

    # Ensure we return at least some function
    if not best_function:
        # Fallback to simple construction
        best_function = [1.0] * 100

    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")