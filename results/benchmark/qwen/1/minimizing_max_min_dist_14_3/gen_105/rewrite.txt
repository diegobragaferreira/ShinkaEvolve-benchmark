# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import random
import warnings

class SphericalVoronoiEvolution:
    """
    Evolutionary optimizer using spherical Voronoi diagrams for 3D point dispersion.
    """
    
    def __init__(self, n_points: int = 14, seed: int = 42):
        self.n_points = n_points
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def fibonacci_sphere(self, samples: int = 14) -> np.ndarray:
        """Generate points on unit sphere using Fibonacci spiral."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def normalize_vector(self, vec):
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-12 else vec
    
    def generate_random_pole(self):
        """Generate random point on unit sphere."""
        # Generate random point in 3D and normalize
        point = np.random.randn(3)
        return self.normalize_vector(point)
    
    def create_voronoi_from_poles(self, poles):
        """Create spherical Voronoi diagram from pole points."""
        try:
            # Create SphericalVoronoi object
            sv = SphericalVoronoi(poles, radius=1.0, center=np.zeros(3))
            return sv
        except:
            # Fallback to simpler approach if SphericalVoronoi fails
            return None
    
    def voronoi_fitness(self, poles):
        """Calculate fitness based on Voronoi diagram properties."""
        try:
            sv = self.create_voronoi_from_poles(poles)
            if sv is None:
                return -np.inf
                
            # Get Voronoi cells
            cells = sv._vertices
            
            # Calculate pairwise distances between poles
            distances = pdist(poles)
            
            if len(distances) == 0:
                return -np.inf
                
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            # Prefer well-distributed poles
            if max_dist > 0:
                ratio = min_dist / max_dist
            else:
                ratio = 0.0
                
            # Add penalty for degenerate cases
            if min_dist < 1e-6:
                ratio -= 1000.0
            
            return ratio
            
        except Exception as e:
            return -np.inf
    
    def mutate_poles(self, poles, mutation_rate=0.1, strength=0.1):
        """Mutate poles with spherical Gaussian noise."""
        mutated = poles.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Add spherical Gaussian noise
                noise = np.random.randn(3) * strength
                # Project noise onto tangent plane and exponentiate
                current = mutated[i]
                # Simple approach: add noise and renormalize
                mutated[i] = self.normalize_vector(current + noise)
        return mutated
    
    def crossover_poles(self, parent1, parent2):
        """Single-point crossover between two pole sets."""
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    def create_initial_population(self, population_size=20):
        """Create initial population of pole configurations."""
        population = []
        for _ in range(population_size):
            # Start with Fibonacci points and add small random noise
            poles = self.fibonacci_sphere(self.n_points)
            # Add small random perturbation
            noise = np.random.normal(0, 0.05, poles.shape)
            poles = poles + noise
            # Renormalize
            poles = np.array([self.normalize_vector(p) for p in poles])
            population.append(poles)
        return population
    
    def evolve_voronoi_population(self, population_size=20, generations=50):
        """Evolve population of Voronoi pole configurations."""
        # Create initial population
        population = self.create_initial_population(population_size)
        fitness_history = []
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for poles in population:
                fitness = self.voronoi_fitness(poles)
                fitness_scores.append(fitness)
            
            # Track best
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            fitness_history.append(best_fitness)
            
            # Print progress
            if generation % 10 == 0:
                print(f"Gen {generation}: Best fitness = {best_fitness:.6f}")
            
            # Selection (tournament selection)
            selected = []
            tournament_size = 3
            for _ in range(population_size):
                tournament_indices = random.sample(range(population_size), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected.append(population[winner_idx].copy())
            
            # Crossover and mutation
            new_population = []
            for i in range(0, population_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % population_size]
                
                # Crossover
                child1, child2 = self.crossover_poles(parent1, parent2)
                
                # Mutation
                child1 = self.mutate_poles(child1, mutation_rate=0.2, strength=0.1)
                child2 = self.mutate_poles(child2, mutation_rate=0.2, strength=0.1)
                
                new_population.extend([child1, child2])
            
            population = new_population[:population_size]
            
            # Early stopping if we're converging
            if len(fitness_history) > 5:
                recent_improvement = fitness_history[-1] - fitness_history[-5]
                if recent_improvement < 1e-6 and generation > 20:
                    break
        
        # Return best solution
        final_fitness_scores = [self.voronoi_fitness(poles) for poles in population]
        best_idx = np.argmax(final_fitness_scores)
        return population[best_idx]
    
    def voronoi_to_points(self, poles):
        """Convert Voronoi poles to actual 3D points."""
        try:
            # Convert Voronoi poles to proper point set
            # Use the poles themselves as the point set
            return poles
        except:
            # Fallback to Fibonacci points
            return self.fibonacci_sphere(self.n_points)
    
    def local_refinement(self, points, max_iter=100):
        """Refine points using local optimization."""
        # Define objective function
        def objective(x_flat):
            points = x_flat.reshape(-1, 3)
            distances = pdist(points)
            if len(distances) == 0:
                return np.inf
            d_min = np.min(distances)
            d_max = np.max(distances)
            if d_max == 0:
                return np.inf
            # Maximize ratio
            return -d_min / d_max
        
        # Local optimization using L-BFGS-B
        flat_points = points.flatten()
        bounds = [(0, 1) for _ in range(len(flat_points))]
        
        try:
            result = minimize(
                objective,
                flat_points,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                # Ensure all points are within [0,1]^3
                refined_points = np.clip(refined_points, 0, 1)
                return refined_points
        except:
            pass
            
        return points
    
    def optimize(self):
        """Main optimization function."""
        # Step 1: Evolutionary optimization with Voronoi
        print("Starting evolutionary optimization...")
        poles = self.evolve_voronoi_population(population_size=20, generations=50)
        
        # Step 2: Convert to points and refine
        points = self.voronoi_to_points(poles)
        
        # Step 3: Local refinement
        print("Applying local refinement...")
        refined_points = self.local_refinement(points, max_iter=200)
        
        # Step 4: Final validation
        distances = pdist(refined_points)
        if len(distances) > 0:
            min_dist = np.min(distances)
            if min_dist < 1e-12:
                # Fall back to initial configuration
                return self.fibonacci_sphere(self.n_points)
        
        return refined_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    try:
        optimizer = SphericalVoronoiEvolution(n_points=14, seed=42)
        return optimizer.optimize()
    except Exception as e:
        # Fallback to basic approach
        warnings.warn(f"Fallback due to error: {str(e)}")
        # Use Fibonacci initialization as last resort
        points = np.array([])
        try:
            # Fibonacci on sphere
            phi = np.pi * (3. - np.sqrt(5.))
            points = []
            for i in range(14):
                y = 1 - (i / float(14 - 1)) * 2
                radius = np.sqrt(1 - y * y)
                theta = phi * i
                x = np.cos(theta) * radius
                z = np.sin(theta) * radius
                points.append([x, y, z])
            points = np.array(points)
            # Normalize to [0,1]^3
            points = points - np.mean(points, axis=0)
            max_coord = np.max(np.abs(points))
            if max_coord > 0:
                points = points / max_coord * 0.5
            points = points + 0.5
        except:
            # Final fallback - random points
            points = np.random.rand(14, 3)
        return points

# EVOLVE-BLOCK-END