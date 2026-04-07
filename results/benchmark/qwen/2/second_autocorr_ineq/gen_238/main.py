# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import math
import random
from scipy.fft import fft, ifft

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autoconvolution_norms(f_values):
    """
    Compute the three norms needed for C2 calculation.
    f_values: list of step heights
    Returns: ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array
    f = np.array(f_values)

    # Perform convolution (autoconvolution)
    # Using 'full' mode to get complete convolution result
    g = signal.convolve(f, f, mode='full')

    # Calculate the norms
    # ||g||₂² = sum(gᵢ²)
    g_squared = g * g
    norm_g_2_squared = np.sum(g_squared)

    # ||g||₁ = sum(|gᵢ|)
    norm_g_1 = np.sum(np.abs(g))

    # ||g||∞ = max(|gᵢ|)
    norm_g_inf = np.max(np.abs(g))

    return norm_g_2_squared, norm_g_1, norm_g_inf

def compute_c2_from_norms(norm_2_sq, norm_1, norm_inf):
    """Helper function to compute C2 from precomputed norms"""
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    return norm_2_sq / (norm_1 * norm_inf)

def adaptive_peak_spacing_construction(n_steps: int = None) -> list[float]:
    """
    Construct function with adaptive peak spacing that targets optimal C2 properties.
    This approach specifically designs peaks to create beneficial autoconvolution characteristics.
    """
    if n_steps is None:
        n_steps = random.randint(800, 2000)

    # Create a function that avoids sharp autoconvolution peaks
    # by spacing the peaks appropriately

    # Create base shape with logarithmic peak distribution
    # This ensures better coverage of the domain and prevents clumping
    n_peaks = max(3, min(20, n_steps // 100))

    # Use logarithmic spacing to distribute peaks
    log_positions = np.logspace(0, 1, n_peaks, base=10) - 1  # [0, 9]
    peak_positions = (log_positions / 9.0) * (n_steps - 10) + 5  # Centered in domain

    # Ensure unique positions
    peak_positions = np.unique(peak_positions.astype(int))
    while len(peak_positions) < n_peaks:
        # Add more positions if needed
        new_pos = np.random.randint(5, n_steps - 5)
        if new_pos not in peak_positions:
            peak_positions = np.append(peak_positions, new_pos)
    peak_positions = np.sort(peak_positions)[:n_peaks]

    # Create base function
    f_values = np.zeros(n_steps)

    # Generate peaks with varying characteristics
    for i, center in enumerate(peak_positions):
        # Width varies based on peak importance and position
        # Outer peaks wider to reduce autoconvolution spike effects
        if i == 0 or i == len(peak_positions) - 1:
            width = max(15, min(80, n_steps // 10))
            height = random.uniform(1.5, 2.5)
        else:
            # Inner peaks narrower and less tall to allow better interaction
            width = max(10, min(50, n_steps // 15))
            height = random.uniform(1.0, 2.0)

        # Apply peak shape with exponential decay
        x = np.arange(n_steps)
        # Use exponential decay instead of Gaussian for potentially better autoconvolution
        # since it produces more predictable shapes
        peak = height * np.exp(-0.5 * ((x - center) / width) ** 2)
        f_values += peak

        # Add some random variation to avoid perfect symmetry
        if i % 2 == 0:
            f_values += 0.1 * height * np.exp(-0.5 * ((x - center - 10) / (width * 0.5)) ** 2)

    # Apply smoothing to reduce extreme variations
    if n_steps > 50:
        window_size = min(51, n_steps - 1)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size > 1:
            f_values = signal.savgol_filter(f_values, window_size, 3)

    # Ensure non-negativity and normalize
    f_values = np.maximum(f_values, 0)
    max_val = np.max(f_values)
    if max_val > 0:
        f_values = f_values / max_val * 2.0

    return f_values.tolist()

def flat_autoconvolution_refinement(f_values: list[float], max_iterations: int = 50) -> list[float]:
    """
    Refine function to produce flatter autoconvolution profiles which typically yield higher C2 values.
    Focuses on reducing sharp peaks in autoconvolution while maintaining spread.
    """
    f_array = np.array(f_values)
    best_f = f_array.copy()
    best_c2 = 0.0

    # Precompute original autoconvolution for comparison
    g_original = signal.convolve(f_array, f_array, mode='full')
    g_original = g_original[len(f_array)-1:2*len(f_array)-1]
    orig_norm_2_sq, orig_norm_1, orig_norm_inf = compute_autoconvolution_norms(f_values)
    orig_c2 = compute_c2_from_norms(orig_norm_2_sq, orig_norm_1, orig_norm_inf)

    if orig_c2 > best_c2:
        best_c2 = orig_c2
        best_f = f_array.copy()

    # Iteratively refine to flatten the autoconvolution profile
    for iteration in range(max_iterations):
        # Try various refinements
        candidate_f = f_array.copy()

        # Apply adaptive modifications to reduce peak dominance
        # This is crucial for improving the ratio ||g||₂² / (||g||₁ · ||g||∞)

        # Try a few different transformations to create flatter autoconvolution
        if random.random() < 0.5:
            # Reduce peak heights slightly and add some smoothing
            reduction_factor = random.uniform(0.9, 1.0)
            candidate_f = candidate_f * reduction_factor

            # Apply a mild low-pass filter to flatten the function
            if len(candidate_f) > 10:
                # Use a moving average filter for simplicity
                window = min(21, len(candidate_f) // 10)
                if window % 2 == 0:
                    window += 1
                if window > 1:
                    # Simple moving average
                    smoothed = np.convolve(candidate_f, np.ones(window)/window, mode='same')
                    candidate_f = smoothed

        # Ensure non-negativity
        candidate_f = np.maximum(candidate_f, 0)

        # Re-evaluate
        try:
            test_norm_2_sq, test_norm_1, test_norm_inf = compute_autoconvolution_norms(candidate_f.tolist())
            test_c2 = compute_c2_from_norms(test_norm_2_sq, test_norm_1, test_norm_inf)

            if test_c2 > best_c2:
                best_c2 = test_c2
                best_f = candidate_f.copy()
        except:
            continue

    return best_f.tolist()

def optimized_frequency_domain_construction(n_steps: int = None) -> list[float]:
    """
    Enhanced frequency domain approach that creates optimal frequency characteristics
    for producing high C2 autoconvolutions.
    """
    if n_steps is None:
        n_steps = random.randint(1000, 2500)

    # Build a carefully constructed frequency spectrum
    frequencies = np.fft.fftfreq(n_steps, 1.0/n_steps)

    # Design spectrum to favor smooth autoconvolution properties:
    # 1. Strong low frequency content (promotes smoothness)
    # 2. Controlled high frequency (to avoid noise)
    # 3. Avoid sharp transitions that could create spikes

    # Base smooth envelope with gradual roll-off
    envelope = np.exp(-0.3 * (frequencies / (n_steps/10))**2) * \
               np.exp(-0.05 * np.abs(frequencies))

    # Add some controlled harmonic content that doesn't create sharp peaks in autoconvolution
    harmonics = 0.1 * np.exp(-0.1 * (frequencies / (n_steps/20))**2) * \
                np.cos(0.3 * frequencies / (n_steps/50))

    combined_envelope = envelope + harmonics

    # Normalize and ensure positivity
    combined_envelope = np.maximum(combined_envelope, 0.01)

    # Generate phases
    phases = np.random.uniform(0, 2*np.pi, n_steps)
    phases[0] = 0  # DC component remains real

    # Apply inverse FFT
    complex_signal = combined_envelope * np.exp(1j * phases)
    spatial_function = np.real(ifft(complex_signal))

    # Apply transformations to improve autoconvolution properties
    f_values = np.abs(spatial_function) + 0.01

    # Apply smoothing
    if n_steps > 50:
        window_size = min(51, n_steps - 1)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size > 1:
            f_values = signal.savgol_filter(f_values, window_size, 3)

    # Normalize
    max_val = np.max(f_values)
    if max_val > 0:
        f_values = f_values / max_val * 2.0

    return f_values.tolist()

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value.
    Uses enhanced optimization strategies targeting the specific mathematical properties of C2.
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Try multiple construction strategies and choose the best
    best_result = []
    best_c2 = 0

    # Strategy 1: Adaptive peak spacing construction (new key approach)
    try:
        spacing_result = adaptive_peak_spacing_construction()
        if spacing_result:
            # Evaluate spacing result
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(spacing_result)
            c2 = compute_c2_from_norms(norm_2_sq, norm_1, norm_inf)
            if c2 > best_c2:
                best_c2 = c2
                best_result = spacing_result
    except Exception as e:
        pass

    # Strategy 2: Enhanced frequency domain construction
    try:
        freq_result = optimized_frequency_domain_construction()
        if freq_result:
            # Evaluate freq result
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(freq_result)
            c2 = compute_c2_from_norms(norm_2_sq, norm_1, norm_inf)
            if c2 > best_c2:
                best_c2 = c2
                best_result = freq_result
    except Exception as e:
        pass

    # Strategy 3: Flat autoconvolution refinement of best result so far
    if best_result:
        try:
            refined_result = flat_autoconvolution_refinement(best_result)
            # Evaluate refined result
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(refined_result)
            c2 = compute_c2_from_norms(norm_2_sq, norm_1, norm_inf)
            if c2 > best_c2:
                best_c2 = c2
                best_result = refined_result
        except Exception as e:
            pass

    # Fallback to basic approach if none work
    if not best_result:
        n_steps = 1000
        # Simple but effective approach: create a bell-curved function with some randomness
        x = np.linspace(-1, 1, n_steps)
        base_shape = np.exp(-x**2 / 2)
        base_shape = 0.8 * (base_shape / np.max(base_shape)) + 0.2
        # Add some randomness to break symmetry
        random_noise = np.random.uniform(-0.1, 0.1, n_steps)
        base_shape = np.maximum(base_shape + random_noise, 0)
        best_result = base_shape.tolist()

    # Final optimization: aggressive refinement focusing on C2 improvement
    if best_result:
        try:
            # Try more aggressive refinement approaches
            final_f = np.array(best_result)

            # Compute initial performance
            orig_norm_2_sq, orig_norm_1, orig_norm_inf = compute_autoconvolution_norms(best_result)
            orig_c2 = compute_c2_from_norms(orig_norm_2_sq, orig_norm_1, orig_norm_inf)

            # Try to improve by modifying in a way that maximizes C2 ratio:
            # We want to increase ||g||₂² while keeping ||g||₁ · ||g||∞ small
            improved_f = final_f.copy()

            # Modify some values to reduce peak dominance
            n_modify = max(10, len(final_f) // 20)
            indices_to_modify = random.sample(range(len(final_f)), min(n_modify, len(final_f)))

            for idx in indices_to_modify:
                if random.random() < 0.7:
                    # Reduce value slightly to flatten autoconvolution peaks
                    improved_f[idx] = max(0, improved_f[idx] * 0.95)
                else:
                    # Slight increase to maintain overall energy
                    improved_f[idx] = max(0, improved_f[idx] * 1.02)

            # Re-evaluate
            test_norm_2_sq, test_norm_1, test_norm_inf = compute_autoconvolution_norms(improved_f.tolist())
            test_c2 = compute_c2_from_norms(test_norm_2_sq, test_norm_1, test_norm_inf)

            if test_c2 > orig_c2:
                best_result = improved_f.tolist()

        except Exception as e:
            pass

    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")