import numpy as np

class PSO:
    def __init__(self, n_particles, lr_range, inertia=0.8, cognitive=1.9, social=1.9):
        """
        Initialize the PSO algorithm for learning rate optimization.
        
        :param n_particles: Number of particles (learning rate candidates)
        :param lr_range: Range of learning rates to explore (e.g., [1e-6, 1e-3])
        :param inertia: Inertia weight to control particle movement (default: 0.5)
        :param cognitive: Cognitive weight (influence of personal best, default: 1.5)
        :param social: Social weight (influence of global best, default: 1.5)
        """
        self.n_particles = n_particles
        self.lr_range = lr_range
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        
        # Initialize positions (learning rates) randomly within the specified range
        self.positions = np.random.uniform(low=lr_range[0], high=lr_range[1], size=(n_particles,))
        # Initialize velocities randomly
        self.velocities = np.random.uniform(-0.1, 0.1, size=(n_particles,))
        # Initialize personal best positions and their corresponding losses
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_losses = np.full(n_particles, np.inf)
        # Global best position and corresponding loss
        self.global_best_position = None
        self.global_best_loss = np.inf

    def update(self):
        """
        Update the learning rates (positions) of the particles based on PSO rules.
        """
        for i in range(self.n_particles):
            r1, r2 = np.random.rand(2)  # Random coefficients for stochastic behavior
    
            # Cognitive and social velocities
            cognitive_velocity = self.cognitive * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_velocity = self.social * r2 * (self.global_best_position - self.positions[i])
    
            # Update the velocity
            self.velocities[i] = self.inertia * self.velocities[i] + cognitive_velocity + social_velocity
    
            # Update the learning rate position based on velocity
            self.positions[i] += self.velocities[i]
    
            # Only clip learning rate if it's far outside the bounds
            if self.positions[i] < self.lr_range[0]:
                self.positions[i] = self.lr_range[0] + np.random.uniform(0, 1e-6)  # Small nudge to avoid staying at the lower bound
            elif self.positions[i] > self.lr_range[1]:
                self.positions[i] = self.lr_range[1] - np.random.uniform(0, 1e-6)  # Small nudge to avoid staying at the upper bound
    
            # Debugging: Print velocities and updated positions
            print(f"Particle {i}: Velocity = {self.velocities[i]}, Updated LR = {self.positions[i]}")



    def evaluate_fitness(self, val_loss, current_particle_index):
        """
        Evaluate the fitness of the current particle based on validation loss.
        
        :param val_loss: The validation loss corresponding to the current learning rate (lower is better)
        :param current_particle_index: The index of the particle whose learning rate was used in the last training epoch
        """
        current_loss = val_loss  # Validation loss for the current particle's learning rate
    
        i = current_particle_index  # The particle that was used in the last epoch
    
        # Update personal best if the current validation loss is better for this particle
        if current_loss < self.personal_best_losses[i]:
            self.personal_best_losses[i] = current_loss  # Update the personal best loss
            self.personal_best_positions[i] = self.positions[i]  # Update the personal best learning rate (position)
            print(f"Particle {i}: New Personal Best LR = {self.personal_best_positions[i]}, Loss = {self.personal_best_losses[i]}", flush=True)
            
        # Update global best if the current validation loss is the best across all particles
        if current_loss < self.global_best_loss:
            self.global_best_loss = current_loss  # Update global best loss
            self.global_best_position = self.positions[i]  # Update global best learning rate (position)
            print(f"New Global Best LR = {self.global_best_position}, Loss = {self.global_best_loss}", flush=True)