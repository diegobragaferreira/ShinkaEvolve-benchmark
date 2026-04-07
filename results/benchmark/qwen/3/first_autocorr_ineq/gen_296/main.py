# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time
import random
from typing import List, Optional, Tuple
import warnings

# Suppress scientific notation for cleaner output
np.set_printoptions(suppress=True)

# Global constants
MAX_TIME_SECONDS = 180
MIN_SEQ_LENGTH = 10
MAX_SEQ_LENGTH = 1000
BENCHMARK_RATIO = 1.5031
IMPROVEMENT_THRESHOLD = 1e-6
MAX_ITERATIONS = 1000
POPULATION_SIZE = 50
INERTIA_WEIGHT = 0.7
COGNITIVE_WEIGHT = 1.5
SOCIAL_WEIGHT = 1.5
VELOCITY_CLAMP = 10.0
POSITION_CLAMP_MIN = 0.0
POSITION_CLAMP_MAX = 1000.0

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

class PSOOptimizer:
    def __init__(self):
        self.swarm = []
        self.personal_best_positions = []
        self.personal_best_values = []
        self.global_best_position = None
        self.global_best_value = float('inf')
        
    def convolve_fft(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute convolution using FFT for efficiency with optimized padding."""
        n = len(a)

        # For very small sequences, use direct convolution for numerical stability
        if n < 10:
            return np.convolve(a, b, mode='full')

        # For medium sequences, use FFT with numerical stability checks
        if n <= 1000:
            # Use next power of 2 for better FFT performance
            fft_size = 1 << (n - 1).bit_length()  # Next power of 2
            if fft_size < 2 * n - 1:
                fft_size *= 2

            # Pad to fft_size
            a_padded = np.pad(a, (0, fft_size - n), 'constant')
            b_padded = np.pad(b, (0, fft_size - n), 'constant')

            # Perform convolution in frequency domain
            a_fft = fft(a_padded)
            b_fft = fft(b_padded)
            conv_result = ifft(a_fft * np.conj(b_fft))

            # Check for numerical issues
            if np.any(np.isnan(conv_result)) or np.any(np.isinf(conv_result)):
                # Fall back to direct method if issues occur
                return np.convolve(a, b, mode='full')

            return np.real(conv_result[:2*n-1])
        else:
            # For very large sequences, use direct convolution to avoid FFT overhead
            return np.convolve(a, b, mode='full')

    def compute_c1_constant(self, sequence: List[float]) -> float:
        """Computes the C1 constant for a given sequence."""
        n = len(sequence)
        if n == 0:
            return float('inf')
        else:
            # Compute convolution using FFT
            conv = self.convolve_fft(np.array(sequence), np.array(sequence))
            max_conv = np.max(conv)
            sum_sq = np.sum(sequence)**2

            if sum_sq < 1e-10:
                return float('inf')
            else:
                c1 = 2 * n * max_conv / sum_sq
                # Return 1/C1 to maximize
                return 1.0 / c1

    def initialize_swarm(self, dimensions: int, population_size: int = POPULATION_SIZE) -> None:
        """Initialize the swarm with random particles."""
        self.swarm = []
        self.personal_best_positions = []
        self.personal_best_values = []
        
        for _ in range(population_size):
            # Initialize particle position within bounds
            position = np.random.uniform(POSITION_CLAMP_MIN, POSITION_CLAMP_MAX, dimensions)
            self.swarm.append(position.copy())
            
            # Initialize velocity
            velocity = np.random.uniform(-VELOCITY_CLAMP, VELOCITY_CLAMP, dimensions)
            
            # Personal best is initially the particle's own position
            self.personal_best_positions.append(position.copy())
            self.personal_best_values.append(self.compute_c1_constant(position.tolist()))
            
            # Update global best if needed
            if self.personal_best_values[-1] > self.global_best_value:
                self.global_best_value = self.personal_best_values[-1]
                self.global_best_position = position.copy()

    def update_velocity_and_position(self, particle_idx: int, iteration: int) -> None:
        """Update velocity and position for a single particle."""
        particle_pos = self.swarm[particle_idx]
        particle_vel = np.zeros_like(particle_pos)
        
        # Get personal best and global best
        pbest_pos = self.personal_best_positions[particle_idx]
        gbest_pos = self.global_best_position
        
        # Update velocity with PSO formula
        r1 = np.random.random(len(particle_pos))
        r2 = np.random.random(len(particle_pos))
        
        cognitive_component = COGNITIVE_WEIGHT * r1 * (pbest_pos - particle_pos)
        social_component = SOCIAL_WEIGHT * r2 * (gbest_pos - particle_pos)
        
        # Calculate inertia weight dynamically
        inertia_weight = INERTIA_WEIGHT * (1.0 - iteration / MAX_ITERATIONS)
        
        particle_vel = inertia_weight * particle_vel + cognitive_component + social_component
        
        # Clamp velocity
        particle_vel = np.clip(particle_vel, -VELOCITY_CLAMP, VELOCITY_CLAMP)
        
        # Update position
        new_pos = particle_pos + particle_vel
        
        # Clamp position to valid range
        new_pos = np.clip(new_pos, POSITION_CLAMP_MIN, POSITION_CLAMP_MAX)
        
        self.swarm[particle_idx] = new_pos
        
    def optimize(self, dimensions: int) -> List[float]:
        """Run the PSO optimization process."""
        start_time = time.time()
        self.initialize_swarm(dimensions)
        
        for iteration in range(MAX_ITERATIONS):
            if time.time() - start_time > MAX_TIME_SECONDS - 5:
                break
                
            for i in range(len(self.swarm)):
                # Update particle position and velocity
                self.update_velocity_and_position(i, iteration)
                
                # Evaluate the new position
                current_value = self.compute_c1_constant(self.swarm[i].tolist())
                
                # Update personal best if needed
                if current_value > self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best_positions[i] = self.swarm[i].copy()
                    
                    # Update global best if needed
                    if current_value > self.global_best_value:
                        self.global_best_value = current_value
                        self.global_best_position = self.swarm[i].copy()
                        
        # Return the best found solution
        return self.global_best_position.tolist()

def search_for_best_sequence() -> List[float]:
    """Main entry point for searching the best sequence."""
    # Initialize optimizer
    optimizer = PSOOptimizer()
    
    # Determine initial sequence length
    initial_length = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
    
    # Run PSO optimization
    best_sequence = optimizer.optimize(initial_length)
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")