# EVOLVE-BLOCK-START
import numpy as np
from scipy.fft import fft, ifft
import random
import time
from typing import List, Tuple
import copy
from joblib import Parallel, delayed
from scipy.signal import fftconvolve
import math

class MultiScaleStepFunctionOptimizer:
    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def clear_cache(self):
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def convolve_fft(self, a: List[float], b: List[float]) -> List[float]:
        """Compute convolution using FFT for better performance."""
        n = len(a)
        if n == 0:
            return []

        # Pad to length 2*n - 1 for full convolution
        padded_length = 2 * n - 1
        fa = fft(a, padded_length)
        fb = fft(b, padded_length)
        result = ifft(fa * fb.conj()).real
        # Return only the valid convolution part
        return result[:n].tolist()

    def compute_c1(self, sequence: List[float]) -> float:
        """Compute the C1 constant for a given sequence."""
        n = len(sequence)
        if n == 0:
            return float('inf')

        sum_a = np.sum(sequence)
        if sum_a < 1e-10:
            return float('inf')

        # Compute convolution using FFT
        conv = self.convolve_fft(sequence, sequence)
        max_conv = np.max(conv)

        # Compute C1 = 2n * max(conv) / (sum(a))^2
        c1 = 2 * n * max_conv / (sum_a ** 2)
        return c1

    def evaluate_fitness(self, sequence: List[float]) -> float:
        """Evaluate fitness as inverse of C1 (higher is better)."""
        c1 = self.compute_c1(sequence)
        if c1 == float('inf') or c1 <= 0:
            return 0.0
        return 1.0 / c1

    def evaluate_multi_objective_fitness(self, sequence: List[float]) -> dict:
        """
        Evaluate multiple objectives for a sequence.
        Returns a dictionary of various fitness metrics.
        """
        n = len(sequence)
        if n == 0:
            return {'c1': float('inf'), 'inv_c1': 0.0, 'autocorr_peak': 0.0, 
                   'autocorr_variance': 0.0, 'mass_concentration': 0.0, 'entropy': 0.0}

        sum_a = np.sum(sequence)
        if sum_a < 1e-10:
            return {'c1': float('inf'), 'inv_c1': 0.0, 'autocorr_peak': 0.0, 
                   'autocorr_variance': 0.0, 'mass_concentration': 0.0, 'entropy': 0.0}

        # Compute convolution using FFT
        conv = self.convolve_fft(sequence, sequence)
        max_conv = np.max(conv)
        autocorr_variance = np.var(conv)
        
        # Compute additional metrics
        mass_concentration = sum_a / max_conv if max_conv > 0 else 0
        entropy = 0.0
        if sum_a > 0:
            p = np.array(sequence) / sum_a
            p = p[p > 0]  # Remove zeros to avoid log(0)
            entropy = -np.sum(p * np.log(p))
            
        c1 = 2 * n * max_conv / (sum_a ** 2)
        inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

        return {
            'c1': c1,
            'inv_c1': inv_c1,
            'autocorr_peak': max_conv,
            'autocorr_variance': autocorr_variance,
            'mass_concentration': mass_concentration,
            'entropy': entropy
        }

    def generate_harmonic_step_function(self, n: int, scale_factor: float = 1.0) -> List[float]:
        """
        Generate a step function using harmonic modulation to reduce autocorrelation peaks.
        """
        # Use harmonic frequencies to create non-periodic patterns
        num_steps = max(3, min(30, int(n // (10 * scale_factor))))
        
        # Harmonic frequency-based step positions
        step_positions = []
        for i in range(num_steps):
            # Use harmonic progression for positions
            freq = 1 + i * 0.7
            pos = int(n * (abs(np.sin(freq * np.pi / 2)) % 1))
            step_positions.append(pos)

        # Sort positions and ensure uniqueness
        step_positions = sorted(set(step_positions))
        while len(step_positions) < num_steps:
            new_pos = random.randint(0, n-1)
            if new_pos not in step_positions:
                step_positions.append(new_pos)
        step_positions = sorted(step_positions[:num_steps])

        # Step heights using harmonic modulation
        step_heights = []
        base_height = 100.0 * scale_factor
        for i in range(len(step_positions)):
            # Apply harmonic modulation to heights
            mod_freq = 2 + i * 0.5
            height_mod = abs(np.sin(mod_freq * np.pi / 4))
            height = base_height * height_mod
            # Add noise to break symmetry
            noise = random.uniform(0.9, 1.1)
            step_heights.append(max(0.01, height * noise))

        # Create final sequence
        sequence = [0.0] * n
        for i, (pos, height) in enumerate(zip(step_positions, step_heights)):
            if i < len(step_positions) - 1:
                end_pos = step_positions[i+1]
            else:
                end_pos = n

            pos = max(0, min(n-1, pos))
            end_pos = max(pos+1, min(n, end_pos))

            if end_pos > pos:
                sequence[pos:end_pos] = [height] * (end_pos - pos)

        return sequence

    def generate_multi_scale_step_functions(self, scales: List[int]) -> List[List[float]]:
        """Generate step functions at multiple scales."""
        sequences = []
        for scale in scales:
            seq = self.generate_harmonic_step_function(scale)
            sequences.append(seq)
        return sequences

    def local_gradient_refine(self, sequence: List[float], steps: int = 10) -> List[float]:
        """Refine a sequence using a local gradient-like approach."""
        refined = sequence.copy()
        for _ in range(steps):
            # Compute current fitness landscape around each point
            fitness_landscape = []
            for i in range(len(refined)):
                # Evaluate small changes around point i
                base_fitness = self.evaluate_fitness(refined)
                best_fitness = base_fitness
                best_val = refined[i]
                
                # Test small variations
                test_values = [refined[i] * 0.9, refined[i], refined[i] * 1.1]
                for val in test_values:
                    temp_seq = refined.copy()
                    temp_seq[i] = max(0.01, val)
                    test_fitness = self.evaluate_fitness(temp_seq)
                    if test_fitness > best_fitness:
                        best_fitness = test_fitness
                        best_val = val
                        
                fitness_landscape.append(best_val)
                
            refined = fitness_landscape
            
        return refined

    def adaptive_constraint_relaxation(self, initial_sequence: List[float], iterations: int = 20) -> List[float]:
        """
        Apply adaptive relaxation of constraints to escape local minima.
        """
        current_seq = initial_sequence.copy()
        best_seq = current_seq.copy()
        best_fitness = self.evaluate_fitness(current_seq)
        
        # Start with strict constraints
        max_relaxation = 0.1
        relaxation_rate = 0.01
        
        for i in range(iterations):
            # Gradually relax constraints
            relaxation_factor = max(0.0, max_relaxation - i * relaxation_rate)
            
            # Apply relaxed mutation
            mutated = current_seq.copy()
            for j in range(len(mutated)):
                if random.random() < 0.1 * (1 - relaxation_factor):
                    # Apply perturbation proportional to relaxation factor
                    noise = random.gauss(0, 0.1 * relaxation_factor)
                    mutated[j] = max(0.01, mutated[j] * (1 + noise))
                    
            mutated_fitness = self.evaluate_fitness(mutated)
            
            if mutated_fitness > best_fitness:
                best_seq = mutated.copy()
                best_fitness = mutated_fitness
                
            current_seq = mutated
            
        return best_seq

    def multi_objective_search(self, max_time_seconds: int = 175) -> List[float]:
        """
        Perform multi-objective search to find optimal step function.
        """
        start_time = time.time()
        best_sequence = None
        best_fitness = 0.0
        
        # Predefined scales for multi-scale search
        scales = [10, 50, 100, 500, 1000]
        
        # Generate initial sequences at different scales
        initial_sequences = self.generate_multi_scale_step_functions(scales)
        
        # Optimize each sequence with adaptive refinement
        for seq in initial_sequences:
            if time.time() - start_time > max_time_seconds:
                break
                
            # Apply constraint relaxation to escape local minima
            refined = self.adaptive_constraint_relaxation(seq, 15)
            
            # Fine-tune with local gradient refinement
            fine_tuned = self.local_gradient_refine(refined, 5)
            
            # Evaluate final fitness
            current_fitness = self.evaluate_fitness(fine_tuned)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_sequence = fine_tuned.copy()
                
        return best_sequence if best_sequence is not None else [1.0] * 100

    def search_for_best_sequence(self) -> List[float]:
        """Main search function for finding optimal sequence."""
        self.clear_cache()
        return self.multi_objective_search()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")