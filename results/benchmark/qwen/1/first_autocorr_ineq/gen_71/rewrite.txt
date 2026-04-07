# EVOLVE-BLOCK-START
import numpy as np
from scipy.fft import fft, ifft
from scipy.optimize import minimize_scalar
from typing import List, Optional, Tuple
import math

# Fixed seed for reproducibility
np.random.seed(42)

class AutocorrelationHybridOptimizer:
    """
    An optimized optimizer for finding step functions that maximize 1/C₁,
    thereby improving the upper bound on the first autocorrelation inequality constant.
    This version uses a hybrid optimization approach combining:
    - Mathematical insight-based initialization
    - Line-search optimization with adaptive gradient descent
    - FFT-based efficient convolution computation
    """

    def __init__(self):
        self.seed = 42
        np.random.seed(self.seed)

    def compute_convolution(self, sequence: np.ndarray) -> np.ndarray:
        """Compute the autoconvolution of the sequence using FFT for efficiency."""
        n = len(sequence)
        # For efficiency, use FFT-based convolution for larger sequences
        if n > 500:
            padded_seq = np.pad(sequence, (0, n - 1), 'constant')
            conv_result = np.real(ifft(fft(padded_seq, 2*n-1) * np.conj(fft(sequence, 2*n-1))))
            return conv_result[:2*n - 1]
        else:
            # For smaller sequences, use direct computation
            conv = np.zeros(2 * n - 1)
            for i in range(n):
                for j in range(n):
                    conv[i + j] += sequence[i] * sequence[j]
            return conv

    def compute_c1(self, sequence: np.ndarray) -> float:
        """Compute C₁ constant for the given sequence."""
        n = len(sequence)
        if n == 0:
            return float('inf')
        
        # Compute convolution
        conv = self.compute_convolution(sequence)
        
        # Calculate C₁
        max_conv = np.max(conv)
        sum_sq = np.sum(sequence) ** 2
        if sum_sq == 0:
            return float('inf')
            
        c1 = 2 * n * max_conv / sum_sq
        return c1

    def compute_inv_c1(self, sequence: np.ndarray) -> float:
        """Compute 1/C₁ for the given sequence."""
        c1 = self.compute_c1(sequence)
        if c1 == 0:
            return float('inf')
        return 1.0 / c1

    def compute_gradient_approx(self, sequence: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Compute approximate gradient of 1/C₁ using finite differences."""
        n = len(sequence)
        grad = np.zeros(n)
        base_inv_c1 = self.compute_inv_c1(sequence)
        
        for i in range(n):
            perturbed = sequence.copy()
            perturbed[i] += epsilon
            perturbed = np.maximum(perturbed, 0)  # Ensure non-negative
            perturbed_inv_c1 = self.compute_inv_c1(perturbed)
            grad[i] = (perturbed_inv_c1 - base_inv_c1) / epsilon
            
        return grad

    def line_search(self, sequence: np.ndarray, direction: np.ndarray, 
                    max_alpha: float = 1.0, tolerance: float = 1e-6) -> float:
        """Perform a line search along the given direction to find an optimal step size."""
        def objective(alpha):
            new_seq = sequence + alpha * direction
            new_seq = np.maximum(new_seq, 0)  # Ensure non-negative
            return -self.compute_inv_c1(new_seq)  # Negative because we are minimizing

        # Try different alpha values
        alphas = np.linspace(0, max_alpha, 100)
        best_alpha = 0.0
        best_obj = objective(0.0)
        
        for alpha in alphas[1:]:  # Skip alpha=0 since it's already evaluated
            obj_val = objective(alpha)
            if obj_val > best_obj:
                best_obj = obj_val
                best_alpha = alpha
                
        return best_alpha if best_alpha > tolerance else 0.0

    def initialize_sequence(self) -> List[float]:
        """Initialize a sequence based on mathematical insights and heuristics."""
        n = np.random.randint(100, 1000)
        # Use a combination of exponential decay and uniform distribution
        # to create a sequence that favors higher weights at earlier positions
        # to mimic structures that might yield better C₁ constants
        sequence = np.random.exponential(scale=1.0, size=n)
        # Normalize and scale
        sequence = sequence / np.sum(sequence) * 100
        return sequence.tolist()

    def optimize_single_point(self, initial_sequence: List[float], 
                              max_iter: int = 1000) -> Tuple[List[float], float]:
        """Optimize a single point using adaptive gradient descent with line search."""
        sequence = np.array(initial_sequence, dtype=float)
        best_sequence = sequence.copy()
        best_inv_c1 = self.compute_inv_c1(sequence)
        
        # Initial learning rate
        lr = 0.01
        
        for iteration in range(max_iter):
            # Compute gradient
            grad = self.compute_gradient_approx(sequence)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
                
            # Direction vector
            direction = -grad / grad_norm
            
            # Perform line search along this direction
            alpha = self.line_search(sequence, direction)
            
            # Update sequence
            sequence += alpha * direction
            sequence = np.maximum(sequence, 0)  # Ensure non-negative
            
            # Evaluate new objective
            new_inv_c1 = self.compute_inv_c1(sequence)
            
            # Update best if improved
            if new_inv_c1 > best_inv_c1:
                best_inv_c1 = new_inv_c1
                best_sequence = sequence.copy()
                
                # Early exit if we beat the benchmark
                if best_inv_c1 > 1.0 / 1.5031:
                    break
                    
            # Adaptive learning rate update
            if iteration > 0 and iteration % 10 == 0:
                # Reduce learning rate if improvement is slow
                lr *= 0.95
                lr = max(lr, 1e-6)
                
        return best_sequence.tolist(), best_inv_c1

    def run_optimization(self, max_trials: int = 10) -> Tuple[List[float], float]:
        """Run the optimization process with multiple trials."""
        best_sequence = None
        best_inv_c1 = -float('inf')
        
        for trial in range(max_trials):
            # Initialize sequence
            initial_seq = self.initialize_sequence()
            
            # Optimize
            final_seq, final_inv_c1 = self.optimize_single_point(initial_seq)
            
            # Track best
            if final_inv_c1 > best_inv_c1:
                best_inv_c1 = final_inv_c1
                best_sequence = final_seq
                
        return best_sequence, best_inv_c1

def search_for_best_sequence() -> list[float]:
    """Main function to search for the best coefficient sequence."""
    optimizer = AutocorrelationHybridOptimizer()
    best_sequence, _ = optimizer.run_optimization()
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")