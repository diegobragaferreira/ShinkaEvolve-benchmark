# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft, ifft
import time
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

class AutocorrelationOptimizer:
    def __init__(self, max_iterations=1000, timeout_seconds=180):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.best_sequence = None
        self.best_inv_c1 = 0.0

    def convolve_fft(self, a, b):
        """Compute convolution using FFT for better performance."""
        n = len(a)
        # Zero-pad to avoid circular convolution effects
        padded_length = 2 * n - 1
        fa = fft(a, padded_length)
        fb = fft(b, padded_length)
        result = ifft(fa * fb).real
        return result[:n]

    def compute_c1(self, sequence):
        """Compute the C1 constant from the sequence."""
        if len(sequence) == 0:
            return float('inf')
        sum_a = np.sum(sequence)
        if sum_a < 1e-10:
            return float('inf')

        convolved = self.convolve_fft(sequence, sequence)
        max_conv = np.max(convolved)
        n = len(sequence)
        c1 = (2 * n * max_conv) / (sum_a ** 2)
        return c1

    def compute_inv_c1(self, sequence):
        """Compute inverse of C1 (the value we want to maximize)."""
        c1 = self.compute_c1(sequence)
        return 1.0 / c1 if c1 > 0 else 0.0

    def quadratic_objective(self, a):
        """Quadratic objective for optimization."""
        sum_a = np.sum(a)
        if sum_a < 1e-10:
            return float('inf')
        
        convolved = self.convolve_fft(a, a)
        max_conv = np.max(convolved)
        
        # Return the ratio we want to minimize: max_conv / (sum_a)^2 
        # We return negative because we're maximizing 1/C1 = (sum_a)^2 / max_conv
        return -max_conv / (sum_a ** 2)

    def optimize_sequence(self):
        """Main optimization loop using quadratic programming."""
        start_time = time.time()
        
        # Generate initial sequence
        n = random.randint(100, 1000)
        current_sequence = [random.uniform(0.1, 1000) for _ in range(n)]
        
        # Convert to numpy array for easier handling
        current_sequence = np.array(current_sequence)
        
        # Optimization using scipy's minimize
        # We use SLSQP method which handles constraints well
        bounds = [(0.01, 1000.0)] * len(current_sequence)
        constraints = [{'type': 'ineq', 'fun': lambda x: np.sum(x) - 0.01}]
        
        # Initial guess
        x0 = current_sequence.copy()
        
        try:
            result = minimize(
                fun=self.quadratic_objective,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success:
                optimized_sequence = result.x
                inv_c1 = self.compute_inv_c1(optimized_sequence)
                if inv_c1 > self.best_inv_c1:
                    self.best_inv_c1 = inv_c1
                    self.best_sequence = optimized_sequence.tolist()
            else:
                # Fallback to simple random search if optimization fails
                pass
                
        except Exception:
            pass
            
        # If optimization didn't improve, fall back to random search
        if self.best_sequence is None:
            self.best_sequence = current_sequence.tolist()
            self.best_inv_c1 = self.compute_inv_c1(self.best_sequence)
            
        # Run some iterations of local search for better results
        for iteration in range(self.max_iterations):
            if time.time() - start_time > self.timeout_seconds:
                break
                
            # Random small perturbations
            new_sequence = current_sequence.copy()
            for i in range(len(new_sequence)):
                if random.random() < 0.1:  # 10% chance to modify
                    new_sequence[i] *= random.uniform(0.9, 1.1)
                    new_sequence[i] = max(0.01, min(1000.0, new_sequence[i]))
                    
            new_inv_c1 = self.compute_inv_c1(new_sequence)
            if new_inv_c1 > self.best_inv_c1:
                self.best_inv_c1 = new_inv_c1
                self.best_sequence = new_sequence.tolist()
                
            current_sequence = new_sequence
            
        return self.best_sequence

def search_for_best_sequence():
    """Function to search for the best coefficient sequence."""
    optimizer = AutocorrelationOptimizer()
    return optimizer.optimize_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")