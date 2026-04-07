# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import convolve
import time
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

class ConvolutionOptimizer:
    """
    A class-based optimizer for computing the optimal step function that maximizes 1/C1.
    """

    def __init__(self, max_time_seconds=170):
        self.max_time_seconds = max_time_seconds

    def compute_c1(self, sequence):
        """Compute the C1 constant for a given sequence."""
        if len(sequence) == 0:
            return float('inf')

        # Use FFT-based convolution for efficiency
        conv = convolve(sequence, sequence, mode='full')
        # Take only the relevant part of convolution (the peak)
        max_conv = np.max(conv[len(sequence)-1:])  # From index n-1 onwards

        # Normalize and compute C1
        sum_sq = np.sum(sequence)**2
        if sum_sq == 0:
            return float('inf')

        c1 = (2 * len(sequence) * max_conv) / sum_sq
        return c1

    def compute_inv_c1(self, sequence):
        """Compute inverse of C1 (what we want to maximize)."""
        c1 = self.compute_c1(sequence)
        if c1 == 0 or np.isinf(c1):
            return 0
        return 1.0 / c1

    def estimate_convolution_gradient(self, sequence, epsilon=1e-6):
        """
        Estimate the gradient of the maximum convolution value with respect to sequence elements.
        This provides a more direct gradient for optimization compared to finite differences.
        """
        n = len(sequence)
        grad = np.zeros(n)
        base_conv = convolve(sequence, sequence, mode='full')
        max_conv_idx = np.argmax(base_conv[len(sequence)-1:])
        max_conv_val = base_conv[len(sequence)-1:][max_conv_idx]
        
        for i in range(n):
            # Perturb the i-th element
            perturbed_seq = sequence.copy()
            perturbed_seq[i] += epsilon
            perturbed_seq = np.maximum(perturbed_seq, 0)  # Ensure non-negative
            
            perturbed_conv = convolve(perturbed_seq, perturbed_seq, mode='full')
            perturbed_max_conv = perturbed_conv[len(sequence)-1:][max_conv_idx]
            
            # Estimate gradient using finite difference
            grad[i] = (perturbed_max_conv - max_conv_val) / epsilon
        
        return grad

    def project_sequence(self, sequence):
        """Ensure sequence meets all constraints."""
        # Ensure all elements are non-negative
        sequence = np.maximum(sequence, 0)
        
        # Ensure sum is at least 0.01 (avoid trivial solutions)
        if np.sum(sequence) < 0.01:
            sequence[0] = 0.1
        
        # Clip all elements to [0, 1000]
        sequence = np.clip(sequence, 0, 1000)
        
        return sequence

    def gradient_ascent_step(self, sequence, learning_rate=0.01, max_iterations=10):
        """Perform gradient ascent step guided by estimated gradients."""
        old_sequence = sequence.copy()
        
        for _ in range(max_iterations):
            # Estimate gradient of convolution peak w.r.t. sequence
            grad = self.estimate_convolution_gradient(sequence)
            
            # Update using gradient ascent
            sequence = sequence + learning_rate * grad
            
            # Project onto feasible set
            sequence = self.project_sequence(sequence)
            
            # Early stopping if change is small
            if np.linalg.norm(sequence - old_sequence) < 1e-6:
                break
                
            old_sequence = sequence.copy()
        
        return sequence

    def generate_initial_sequence(self, length=None):
        """Generate a good initial sequence based on known patterns."""
        if length is None:
            length = random.randint(100, 500)
        
        # Use exponential decay pattern which often performs well
        decay_factor = 0.95
        sequence = [1.0 * (decay_factor ** i) for i in range(length)]
        
        # Ensure minimum values
        sequence = [max(x, 0.01) for x in sequence]
        
        # Add slight noise to avoid degeneracy
        noise_factor = 0.05
        sequence = [x * (1 + random.uniform(-noise_factor, noise_factor)) for x in sequence]
        sequence = [max(x, 0.01) for x in sequence]
        
        return sequence

    def local_search_refinement(self, sequence, iterations=20):
        """Enhanced local refinement using gradient ascent."""
        current_seq = np.array(sequence)
        
        for _ in range(iterations):
            # Perform gradient ascent update
            current_seq = self.gradient_ascent_step(current_seq, learning_rate=0.005)
            
            # Occasionally apply heuristic adjustments
            if random.random() < 0.1:  # 10% chance
                # Flatten high convolution regions
                conv = convolve(current_seq, current_seq, mode='full')
                conv_part = conv[len(current_seq)-1:]
                max_conv = np.max(conv_part)
                
                # Identify and reduce contributions to peak convolution
                max_indices = np.where(conv_part >= 0.9 * max_conv)[0]
                for idx in max_indices[:min(3, len(max_indices))]:
                    for offset in [-1, 0, 1]:
                        pos = idx + offset
                        if 0 <= pos < len(current_seq):
                            current_seq[pos] *= 0.98
        
        # Final projection
        current_seq = self.project_sequence(current_seq)
        
        return current_seq.tolist()

    def multi_scale_optimization(self, initial_sequence, max_iter=30):
        """Perform optimization at multiple scales to avoid local optima."""
        sequence = np.array(initial_sequence)
        best_inv_c1 = self.compute_inv_c1(sequence)
        best_sequence = sequence.copy()
        
        # Coarse scale: optimize with larger steps
        coarse_lr = 0.05
        for _ in range(5):
            sequence = self.gradient_ascent_step(sequence, learning_rate=coarse_lr, max_iterations=5)
            inv_c1 = self.compute_inv_c1(sequence)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = sequence.copy()
        
        # Fine scale: optimize with smaller steps
        fine_lr = 0.005
        for _ in range(10):
            sequence = self.gradient_ascent_step(sequence, learning_rate=fine_lr, max_iterations=2)
            inv_c1 = self.compute_inv_c1(sequence)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = sequence.copy()
        
        # Final adaptive refinement
        adaptive_lr = 0.01
        for iteration in range(max_iter):
            sequence = self.gradient_ascent_step(sequence, learning_rate=adaptive_lr, max_iterations=1)
            inv_c1 = self.compute_inv_c1(sequence)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = sequence.copy()
            # Decrease learning rate over time for stability
            adaptive_lr *= 0.99
        
        return best_sequence.tolist()

    def search_for_best_sequence(self):
        """Main search function using gradient ascent with multi-scale optimization."""
        start_time = time.time()
        best_inv_c1 = 0
        best_sequence = None
        
        # Try multiple initialization strategies
        for attempt in range(10):
            if time.time() - start_time > self.max_time_seconds:
                break
                
            # Initialize with a good pattern
            n = random.randint(100, 500)
            sequence = self.generate_initial_sequence(n)
            
            # Multi-scale optimization
            optimized_sequence = self.multi_scale_optimization(sequence, max_iter=20)
            
            # Evaluate
            inv_c1 = self.compute_inv_c1(optimized_sequence)
            
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = optimized_sequence[:]
        
        # If no good sequence found, fallback to a default initialization
        if best_sequence is None:
            best_sequence = self.generate_initial_sequence(200)
            best_sequence = self.multi_scale_optimization(best_sequence)
        
        return best_sequence

# Main execution function
def search_for_best_sequence():
    """Function to search for the best coefficient sequence."""
    optimizer = ConvolutionOptimizer()
    return optimizer.search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")