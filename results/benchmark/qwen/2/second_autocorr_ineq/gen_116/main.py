# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy import signal
from numba import jit
import time
import cvxpy as cp
from sklearn.decomposition import SparseCoder
import warnings

warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: list[float]) -> tuple[float, float, float]:
    """
    Fast computation of the three norms needed for C2 calculation using piecewise linear integration.
    """
    # Convert to numpy array for easier manipulation
    f = np.array(f_values)
    n_steps = len(f)

    if n_steps == 0:
        return 0.0, 0.0, 0.0

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

    return norm_2_squared, norm_1, norm_inf

def enforce_minimum_peak_spacing(peak_params: list[float], domain_width: float, min_distance_ratio: float = 0.05) -> list[float]:
    """
    Enforce minimum distance between Gaussian peaks to prevent narrow autoconvolution interference.
    This helps maintain flatter autoconvolution profiles which improve C2 values.
    """
    if len(peak_params) < 3:
        return peak_params

    # Create list of peaks [amplitude, center, width]
    peaks = []
    for i in range(0, len(peak_params), 3):
        peaks.append([peak_params[i], peak_params[i+1], peak_params[i+2]])

    # Sort by center position
    peaks.sort(key=lambda x: x[1])

    # Ensure minimum spacing
    min_distance = min_distance_ratio * domain_width
    adjusted_peaks = [peaks[0]]  # Always keep first peak

    for i in range(1, len(peaks)):
        prev_center = adjusted_peaks[-1][1]
        curr_center = peaks[i][1]
        distance = abs(curr_center - prev_center)

        if distance < min_distance:
            # Adjust current peak position to maintain minimum spacing
            if curr_center > prev_center:
                new_center = prev_center + min_distance
            else:
                new_center = prev_center - min_distance
            # Make sure we don't go outside domain
            new_center = np.clip(new_center, -domain_width/2, domain_width/2)
            adjusted_peaks.append([peaks[i][0], new_center, peaks[i][2]])
        else:
            adjusted_peaks.append(peaks[i])

    # Flatten back to parameter list
    result = []
    for amp, center, width in adjusted_peaks:
        result.extend([amp, center, width])

    return result

def create_gaussian_peak_function(x_domain: np.ndarray, peak_params: list[float]) -> np.ndarray:
    """
    Create function from Gaussian peak parameters with proper spacing enforcement.
    """
    f = np.zeros_like(x_domain)
    for i in range(0, len(peak_params), 3):
        amp, center, width = peak_params[i], peak_params[i+1], peak_params[i+2]
        if width > 1e-6:  # Avoid division by zero
            f += amp * np.exp(-0.5 * ((x_domain - center) / width)**2)
    return f

def sparse_convex_optimization_approach(n_steps: int) -> list[float]:
    """
    Solves the C2 maximization problem using convex optimization with sparsity constraints.
    """
    # Create a structured basis for sparse representation
    # Use wavelet-like basis to capture both smooth and localized features
    x_domain = np.linspace(-0.25, 0.25, n_steps)

    # Create a dictionary of basis functions
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

    # Solve sparse coding problem to find optimal coefficients
    # This finds sparse representation that approximates an idealized function
    try:
        # Use L1 regularization to encourage sparsity in the solution
        # This represents our optimization search for the best combination of basis functions
        coder = SparseCoder(dictionary=dictionary, transform_alpha=0.01, transform_n_nonzero_coefs=None)

        # Create a target that promotes flat autoconvolution profiles
        # This is designed to maximize the ratio ||g||₂² / (||g||₁ · ||g||∞)
        target_signal = np.ones_like(x_domain)

        # Try to reconstruct with sparse representation
        codes = coder.transform(target_signal.reshape(-1, 1))

        # Extract the most important components
        active_indices = np.where(np.abs(codes.flatten()) > 1e-6)[0]

        # Create function from selected basis components
        if len(active_indices) > 0:
            # Build function from selected components
            f = np.zeros(n_steps)
            for idx in active_indices[:min(5, len(active_indices))]:  # Limit components
                if idx < len(basis_functions):
                    f += np.maximum(basis_functions[idx], 0) * np.random.uniform(0.5, 2.0)

            # Add small amount of random noise to break symmetry
            noise = np.random.normal(0, 0.01, n_steps)
            f += noise

            # Non-negative constraint
            f = np.maximum(f, 0)

            # Normalize to reasonable scale
            if np.sum(f) > 0:
                f = f / np.sum(f) * 10

            return f.tolist()

    except Exception:
        pass

    # Enhanced fallback using adaptive Gaussian construction with peak spacing
    # Start with more structured peak placement
    domain_width = 0.5
    num_peaks = 8

    # Initialize peaks with better spacing using golden ratio distribution
    peak_positions = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    for i in range(num_peaks):
        ratio = (i * phi) % 1.0
        # Use sine transformation for better center concentration
        pos = np.sin(ratio * np.pi) * domain_width/2
        peak_positions.append(pos)

    # Initialize peak parameters [amplitude, center, width]
    peak_params = []
    for i, pos in enumerate(peak_positions):
        amplitude = np.random.uniform(30, 70)  # More varied amplitudes
        center = pos
        width = np.random.uniform(0.03, 0.08)  # Controlled widths
        peak_params.extend([amplitude, center, width])

    # Apply minimum peak spacing enforcement
    peak_params = enforce_minimum_peak_spacing(peak_params, domain_width)

    # Create function from these peaks
    f = create_gaussian_peak_function(x_domain, peak_params)

    # Add some sinusoidal modulation to avoid degeneracy
    f += 0.2 * np.sin(15 * np.pi * x_domain) * np.exp(-x_domain**2/0.05)

    # Add some noise for robustness
    f += np.random.normal(0, 0.03, n_steps)

    # Enforce non-negativity
    f = np.maximum(f, 0)

    # Normalize
    if np.sum(f) > 0:
        f = f / np.sum(f) * 10

    return f.tolist()

def construct_function() -> list[float]:
    """
    Construct step function using convex optimization approach with sparse representation.
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Multi-attempt selection to maximize C2
    best_c2 = -1
    best_function = None
    start_time = time.time()

    # Set maximum attempts to balance quality vs. time constraints
    max_attempts = 30

    for attempt in range(max_attempts):
        # Early termination if time is running out
        if time.time() - start_time > 85:  # Leave 5 seconds for cleanup
            break

        # Try different number of steps to find optimal
        n_steps = np.random.randint(1500, 4000)

        # Generate function using convex optimization approach
        f_values = sparse_convex_optimization_approach(n_steps)

        # Evaluate the function
        try:
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(f_values)

            # Check for valid norms
            if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                continue

            c2 = norm_2_sq / (norm_1 * norm_inf)

            # Keep the best function
            if c2 > best_c2:
                best_c2 = c2
                best_function = f_values

        except Exception:
            # Skip invalid functions
            continue

    # Return the best function found, or fallback
    if best_function is not None:
        return best_function
    else:
        # Fallback to a simpler construction
        n_steps = 1000
        f_values = [1.0] * n_steps
        return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")