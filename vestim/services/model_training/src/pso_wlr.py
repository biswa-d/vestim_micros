import numpy as np
import torch

class PSOWeightsAndLR:
    def __init__(self, n_particles, model, lr_range, inertia=0.8, cognitive=1.9, social=1.9):
        """
        Initialize the PSO algorithm for both weights and learning rate optimization.
        
        :param n_particles: Number of particles (weight + learning rate candidates)
        :param model: The PyTorch model whose weights will be optimized
        :param lr_range: Range of learning rates to explore (e.g., [1e-7, 1e-5])
        :param inertia: Inertia weight to control particle movement (default: 0.5)
        :param cognitive: Cognitive weight (influence of personal best, default: 1.5)
        :param social: Social weight (influence of global best, default: 1.5)
        """
        self.n_particles = n_particles
        self.model = model
        self.lr_range = lr_range
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        # Initialize positions (weights + learning rates) randomly within the specified range
        self.positions = [
            {
                'weights': self.get_random_weights(),
                'lr': np.random.uniform(low=lr_range[0], high=lr_range[1])
            }
            for _ in range(n_particles)
        ]
        
        # Initialize velocities randomly for weights and learning rates
        self.velocities = [
            {
                'weights': self.get_random_weights(),  # Same structure as weights
                'lr': np.random.uniform(low=lr_range[0], high=lr_range[1])
            }
            for _ in range(n_particles)
        ]
        
        # Initialize personal best positions (weights + learning rates) and their corresponding losses
        self.personal_best_positions = [pos.copy() for pos in self.positions]
        self.personal_best_losses = np.full(n_particles, np.inf)
        
        # Global best position and corresponding loss
        self.global_best_position = None
        self.global_best_loss = np.inf

    def get_random_weights(self):
        """
        Generate random weights for the model (same shape as model weights).
        """
        random_weights = []
        for param in self.model.parameters():
            random_weights.append(torch.randn_like(param).cpu().numpy())
        return random_weights

    def set_model_weights(self, particle):
        """
        Set the model's weights based on a particle's weight positions.
        """
        with torch.no_grad():
            for param, weight in zip(self.model.parameters(), particle['weights']):
                param.copy_(torch.tensor(weight))
        print("Model weights updated")

    def update(self):
        """
        Update the weights and learning rates (positions) of the particles based on PSO rules.
        """
        for i in range(self.n_particles):
            r1, r2 = np.random.rand(2)  # Random coefficients for stochastic behavior

            # Iterate over each parameter in the model
            for param_index, param in enumerate(self.model.parameters()):
                # Convert the current particle's weight and velocity for this parameter to tensors
                current_weights = torch.tensor(self.positions[i]['weights'][param_index])
                current_velocities = torch.tensor(self.velocities[i]['weights'][param_index])

                # Cognitive and social velocities for weights
                cognitive_velocity_weights = self.cognitive * r1 * (torch.tensor(self.personal_best_positions[i]['weights'][param_index]) - current_weights)
                social_velocity_weights = self.social * r2 * (torch.tensor(self.global_best_position['weights'][param_index]) - current_weights)

                # Update the velocities for this parameter
                updated_velocity = self.inertia * current_velocities + cognitive_velocity_weights + social_velocity_weights

                # Update the positions (weights) for this parameter
                updated_position = current_weights + updated_velocity

                # Update the velocity and position in the particle
                self.velocities[i]['weights'][param_index] = updated_velocity.numpy()  # Convert back to numpy
                self.positions[i]['weights'][param_index] = updated_position.numpy()  # Convert back to numpy

                # Apply the updated weights to the model
                param.data.copy_(updated_position)

            # Cognitive and social velocities for learning rate
            cognitive_velocity_lr = self.cognitive * r1 * (self.personal_best_positions[i]['lr'] - self.positions[i]['lr'])
            social_velocity_lr = self.social * r2 * (self.global_best_position['lr'] - self.positions[i]['lr'])

            # Update the learning rate velocity and position
            self.velocities[i]['lr'] = self.inertia * self.velocities[i]['lr'] + cognitive_velocity_lr + social_velocity_lr
            self.positions[i]['lr'] += self.velocities[i]['lr']

            # Clip the learning rate within the specified range
            if self.positions[i]['lr'] < self.lr_range[0]:
                self.positions[i]['lr'] = self.lr_range[0]
            elif self.positions[i]['lr'] > self.lr_range[1]:
                self.positions[i]['lr'] = self.lr_range[1]

            # Debugging: Print velocities and updated positions
            # print(f"Particle {i}: Velocity (weights) = {self.velocities[i]['weights']}, Updated LR = {self.positions[i]['lr']}")


    def evaluate_fitness(self, val_loss, current_particle_index, model):
        """
        Evaluate the fitness of the current particle based on validation loss and save the corresponding weights.
        
        :param val_loss: The validation loss corresponding to the current weights and learning rate.
        :param current_particle_index: The index of the particle used in the last training epoch.
        :param model: The model whose current weights are to be evaluated and potentially saved.
        """
        current_loss = val_loss  # Validation loss for the current particle's weights and learning rate
        
        # Update personal best if the current validation loss is better for this particle
        if current_loss < self.personal_best_losses[current_particle_index]:
            # Save the current best weights from the model
            self.personal_best_positions[current_particle_index]['weights'] = [
                param.detach().cpu().numpy() for param in model.parameters()
            ]
            self.personal_best_positions[current_particle_index]['lr'] = self.positions[current_particle_index]['lr']  # Keep the learning rate as well
            self.personal_best_losses[current_particle_index] = current_loss  # Update the personal best loss
            
            print(f"Particle {current_particle_index}: New Personal Best (LR = {self.personal_best_positions[current_particle_index]['lr']}), Loss = {self.personal_best_losses[current_particle_index]}")
        
        # Update global best if the current validation loss is the best across all particles
        if current_loss < self.global_best_loss:
            # Save the current global best weights and learning rate
            self.global_best_position = self.positions[current_particle_index].copy()
            self.global_best_position['weights'] = [
                param.detach().cpu().numpy() for param in model.parameters()
            ]
            self.global_best_loss = current_loss
            
            print(f"New Global Best (LR = {self.global_best_position['lr']}), Loss = {self.global_best_loss}")

