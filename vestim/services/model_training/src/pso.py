import numpy as np

class PSO:
    def __init__(self, n_particles, lr_range, inertia=0.5, cognitive=1.5, social=1.5):
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
            # Update velocities using the inertia, cognitive, and social components
            r1, r2 = np.random.rand(2)  # Random coefficients for stochastic behavior
            cognitive_velocity = self.cognitive * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_velocity = self.social * r2 * (self.global_best_position - self.positions[i])
            self.velocities[i] = self.inertia * self.velocities[i] + cognitive_velocity + social_velocity
            
            # Update position (learning rate) based on velocity
            self.positions[i] += self.velocities[i]
            # Ensure learning rates stay within the defined range
            self.positions[i] = np.clip(self.positions[i], self.lr_range[0], self.lr_range[1])

    def evaluate_fitness(self, val_loss, current_particle_index):
        """
        Evaluate the fitness of the current particle based on validation loss.
        
        :param val_loss: The validation loss corresponding to the current learning rate (lower is better)
        :param current_particle_index: The index of the particle whose learning rate was used in the last training epoch
        """
        # Update personal best for the specific particle used in the last epoch
        current_loss = val_loss  # Validation loss for the current particle's learning rate

        i = current_particle_index  # The particle that was used in the last epoch

        # Update personal best if the current validation loss is better for this particle
        if current_loss < self.personal_best_losses[i]:
            self.personal_best_losses[i] = current_loss  # Update the personal best loss
            self.personal_best_positions[i] = self.positions[i]  # Update the personal best learning rate (position)
            
        # Update global best if the current validation loss is the best across all particles
        if current_loss < self.global_best_loss:
            self.global_best_loss = current_loss  # Update global best loss
            self.global_best_position = self.positions[i]  # Update global best learning rate (position)
