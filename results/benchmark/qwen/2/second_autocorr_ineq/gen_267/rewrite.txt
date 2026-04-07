# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from joblib import Parallel, delayed
import random
from typing import List, Tuple
import time
from collections import deque
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, ifft
import itertools

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

class SpectralGuidedEvaluator:
    """Specialized evaluator that computes norms and incorporates spectral insights"""
    
    @staticmethod
    def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
        """
        Compute the autoconvolution g = f*f and its norms efficiently.
        Returns (||g||₂², ||g||₁, ||g||∞)
        """
        if not f_values or len(f_values) < 2:
            return 0.0, 0.0, 0.0

        # Create step function on [-1/4, 1/4] with equal spacing
        n = len(f_values)
        
        # Step size in x domain [-1/4, 1/4]
        dx = 0.5 / (n - 1) if n > 1 else 0.5

        # Compute autoconvolution using numpy's convolution
        g = signal.convolve(f_values, f_values, mode='full')
        
        # Extract the central portion representing the actual convolution on [-1/2, 1/2]
        # For two functions of length n on [-1/4, 1/4], convolution produces 2*n-1 points
        center_start = len(g) // 2 - (n - 1)
        center_end = center_start + (2 * n - 1)
        g = g[center_start:center_end]

        # Compute the three norms
        # ||g||∞ = max of |g|
        norm_inf = np.max(np.abs(g)) if len(g) > 0 else 0.0

        # ||g||₁ = sum of |g| * dx
        norm_1 = np.sum(np.abs(g)) * dx if len(g) > 1 else 0.0

        # ||g||₂² = ∫ g² dx using trapezoidal-like integration
        if len(g) <= 1:
            norm_2_squared = 0.0
        else:
            # Use piecewise linear integration for g^2
            # ∫ y^2 dx ≈ (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
            norm_2_squared = 0.0
            for i in range(len(g)-1):
                y1, y2 = g[i], g[i+1]
                norm_2_squared += (dx / 3.0) * (y1**2 + y1*y2 + y2**2)

        return norm_2_squared, norm_1, norm_inf

    @classmethod
    def compute_c2(cls, f_values: List[float]) -> float:
        """Compute the C2 value for given step function."""
        norm_2_squared, norm_1, norm_inf = cls.compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
        
        c2 = norm_2_squared / (norm_1 * norm_inf)
        return c2

class SpectralGuidedOptimizer:
    """New approach using spectral guidance for step function optimization"""
    
    def __init__(self, max_time_seconds: int = 85):
        self.max_time_seconds = max_time_seconds
        self.evaluator = SpectralGuidedEvaluator()
        self.best_solution = None
        self.best_c2 = 0.0
        self.start_time = None
        
    def get_spectrum_info(self, f_values: List[float]) -> Tuple[np.ndarray, float, float]:
        """Get spectral information about a function"""
        if len(f_values) < 2:
            return np.array([0]), 0.0, 0.0
            
        # Get power spectrum (FFT magnitude squared)
        spectrum = np.abs(fft(np.array(f_values))) ** 2
        # Mean and std of spectrum (normalized)
        spectrum_mean = np.mean(spectrum)
        spectrum_std = np.std(spectrum)
        return spectrum, spectrum_mean, spectrum_std
    
    def create_target_spectrum(self, target_frequency_content: str = "flat") -> np.ndarray:
        """Create a target spectral profile"""
        # Create a target spectrum that favors flat autoconvolutions
        n = 100  # Base length for spectrum
        if target_frequency_content == "flat":
            # Flat spectrum suggests evenly distributed energy
            target_spectrum = np.ones(n)
        elif target_frequency_content == "low_freq":
            # Low frequency dominant spectrum
            target_spectrum = np.exp(-np.linspace(0, 3, n)**2)
        else:
            # Default to flat
            target_spectrum = np.ones(n)
        return target_spectrum
    
    def generate_initial_candidate(self, length: int = 500) -> List[float]:
        """Generate an initial candidate using spectral-guided approach"""
        # Use a combination of Gaussian peaks with adaptive parameters
        # based on spectral guidance principles
        
        # Create x coordinates from -1/4 to 1/4
        x = np.linspace(-0.25, 0.25, length)
        
        # Determine number of peaks
        num_peaks = max(3, min(20, length // 50))
        
        # Generate logarithmically spaced peak positions
        log_positions = np.logspace(np.log10(0.05), np.log10(0.45), num_peaks, endpoint=True)
        
        # Initialize function
        individual = np.zeros(length)
        
        # Add peaks with strategic parameters
        for i, log_pos in enumerate(log_positions):
            # Alternate sides for better distribution
            side = (-1) ** i
            base_pos = side * log_pos
            
            # Add small variation
            position_variation = np.random.uniform(-0.005, 0.005) * log_pos
            peak_pos = base_pos + position_variation
            
            # Ensure peak stays within reasonable domain
            if abs(peak_pos) <= 0.24:
                # Distance from center
                center_distance = abs(peak_pos)
                
                # Amplitude based on distance from center
                if center_distance < 0.05:
                    peak_amplitude = np.random.uniform(1.5, 2.5)
                elif center_distance < 0.15:
                    peak_amplitude = np.random.uniform(1.2, 2.0)
                else:
                    peak_amplitude = np.random.uniform(0.8, 1.5)
                
                # Width that promotes flatter autoconvolution
                if center_distance < 0.05:
                    peak_width = np.random.uniform(0.01, 0.025)
                elif center_distance < 0.15:
                    peak_width = np.random.uniform(0.015, 0.035)
                else:
                    peak_width = np.random.uniform(0.02, 0.04)
                
                # Create Gaussian peak
                gaussian_peak = peak_amplitude * np.exp(-0.5 * ((x - peak_pos) / peak_width)**2)
                individual += gaussian_peak
        
        # Ensure non-negative values
        individual = np.clip(individual, 0, None)
        
        # Apply smoothing to reduce sharp transitions
        if length > 50:
            individual = gaussian_filter1d(individual, sigma=0.8)
        
        # Normalize
        if np.max(individual) > 1e-6:
            individual = individual / np.max(individual) * 2.5
            
        return individual.tolist()
    
    def adaptive_directional_search(self, current_solution: List[float], 
                                  search_radius: float = 0.1) -> List[float]:
        """Perform directional search guided by spectral information"""
        n = len(current_solution)
        new_solution = np.array(current_solution)
        
        # Sample directions in function space
        num_directions = min(20, n // 2)
        
        # Create perturbation directions based on current function shape
        # Use derivative-like approach to guide search
        if n > 10:
            # Create a smooth version for direction estimation
            smooth_func = gaussian_filter1d(current_solution, sigma=2)
            
            # Generate multiple random perturbations
            for _ in range(num_directions):
                # Generate direction vector (with preference for smoother changes)
                direction = np.random.randn(n) * search_radius
                
                # Add some correlation based on local function behavior
                if len(smooth_func) > 1:
                    # Make direction somewhat aligned with smooth_func gradient
                    grad = np.gradient(smooth_func)
                    # Add some weight towards gradient direction
                    direction = 0.3 * direction + 0.7 * grad * np.random.rand()
                
                # Apply perturbation
                trial = new_solution + direction
                
                # Clip to non-negative values
                trial = np.clip(trial, 0, None)
                
                # Evaluate trial
                trial_c2 = self.evaluator.compute_c2(trial.tolist())
                
                # Accept if better
                if trial_c2 > self.best_c2:
                    new_solution = trial
                    self.best_c2 = trial_c2
                    self.best_solution = new_solution.tolist()
        
        # Add some additional refinement
        # Small random perturbations to escape local optima
        small_perturb = np.random.randn(n) * 0.05
        trial = new_solution + small_perturb
        trial = np.clip(trial, 0, None)
        
        trial_c2 = self.evaluator.compute_c2(trial.tolist())
        if trial_c2 > self.best_c2:
            new_solution = trial
            self.best_c2 = trial_c2
            self.best_solution = new_solution.tolist()
        
        return new_solution.tolist()
    
    def landscape_mapping(self, candidate: List[float], 
                         resolution: int = 50) -> float:
        """Map the 'landscape' of potential improvements around a candidate"""
        # Create a simple landscape estimate by sampling nearby solutions
        n = len(candidate)
        landscape_score = 0.0
        
        # Sample a few nearby solutions
        for _ in range(10):
            # Perturb with small random changes
            perturbation = np.random.randn(n) * 0.05
            trial = np.array(candidate) + perturbation
            trial = np.clip(trial, 0, None)
            
            c2 = self.evaluator.compute_c2(trial.tolist())
            landscape_score += c2
        
        return landscape_score / 10.0 if 10 > 0 else 0.0
    
    def multi_scale_refinement(self, initial_solution: List[float]) -> List[float]:
        """Refine solution through multi-scale optimization"""
        current_solution = np.array(initial_solution)
        current_c2 = self.evaluator.compute_c2(initial_solution)
        
        if current_c2 > self.best_c2:
            self.best_c2 = current_c2
            self.best_solution = initial_solution.copy()
            
        # Multi-scale approach
        scales = [0.2, 0.1, 0.05]
        for scale in scales:
            # Perform search at this scale
            refined = self.adaptive_directional_search(current_solution.tolist(), 
                                                     search_radius=scale)
            refined_c2 = self.evaluator.compute_c2(refined)
            
            if refined_c2 > current_c2:
                current_solution = np.array(refined)
                current_c2 = refined_c2
                
                if current_c2 > self.best_c2:
                    self.best_c2 = current_c2
                    self.best_solution = current_solution.tolist()
                    
            # Refine further with smaller scale
            if scale > 0.01:
                # Additional directional search with very small radius
                fine_search = self.adaptive_directional_search(current_solution.tolist(), 
                                                             search_radius=scale * 0.5)
                fine_search_c2 = self.evaluator.compute_c2(fine_search)
                if fine_search_c2 > current_c2:
                    current_solution = np.array(fine_search)
                    current_c2 = fine_search_c2
                    
                    if current_c2 > self.best_c2:
                        self.best_c2 = current_c2
                        self.best_solution = current_solution.tolist()
        
        return current_solution.tolist()
    
    def optimize(self) -> List[float]:
        """Main optimization routine using spectral-guided approach"""
        self.start_time = time.time()
        
        # Initial solution generation
        initial_candidates = []
        for _ in range(10):  # Generate multiple initial candidates
            candidate = self.generate_initial_candidate(np.random.randint(500, 2000))
            c2 = self.evaluator.compute_c2(candidate)
            initial_candidates.append((candidate, c2))
            
        # Select best among initial candidates
        if initial_candidates:
            initial_candidates.sort(key=lambda x: x[1], reverse=True)
            best_initial = initial_candidates[0][0]
            self.best_c2 = initial_candidates[0][1]
            self.best_solution = best_initial.copy()
        else:
            # Fallback to simple random generation
            n = np.random.randint(500, 2000)
            self.best_solution = [np.random.random() for _ in range(n)]
            self.best_c2 = self.evaluator.compute_c2(self.best_solution)
        
        # Multi-scale refinement of best initial solution
        refined = self.multi_scale_refinement(self.best_solution)
        refined_c2 = self.evaluator.compute_c2(refined)
        if refined_c2 > self.best_c2:
            self.best_c2 = refined_c2
            self.best_solution = refined
        
        # Continue with directional refinement for remaining time
        iteration = 0
        while (time.time() - self.start_time) < self.max_time_seconds - 5:
            iteration += 1
            if iteration % 10 == 0:
                # Periodic restart with new candidate to avoid stagnation
                new_candidate = self.generate_initial_candidate(np.random.randint(500, 2000))
                new_candidate_c2 = self.evaluator.compute_c2(new_candidate)
                if new_candidate_c2 > self.best_c2:
                    self.best_c2 = new_candidate_c2
                    self.best_solution = new_candidate
            
            # Directional search around current best
            improved = self.adaptive_directional_search(self.best_solution, search_radius=0.1)
            improved_c2 = self.evaluator.compute_c2(improved)
            
            if improved_c2 > self.best_c2:
                self.best_c2 = improved_c2
                self.best_solution = improved
                
        # Final check for any possible improvements
        if self.best_solution:
            # Attempt one final multi-scale refinement
            final_refined = self.multi_scale_refinement(self.best_solution)
            final_c2 = self.evaluator.compute_c2(final_refined)
            if final_c2 > self.best_c2:
                self.best_c2 = final_c2
                self.best_solution = final_refined
                
        return self.best_solution if self.best_solution is not None else []

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value - entry point"""
    try:
        optimizer = SpectralGuidedOptimizer(max_time_seconds=85)
        f_values = optimizer.optimize()
        return f_values
    except Exception as e:
        # Fallback to random generation if anything fails
        print(f"Error in optimization: {e}")
        f_values = [np.random.random()] * np.random.randint(100, 1000)
        return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")