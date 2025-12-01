from typing import Optional


class Logger:

    def __init__(self, model_obj, trainer_obj):

        self.name = "log.txt"
        self.model_obj = model_obj
        self.trainer_obj = trainer_obj
        self.avg_cv_loss: Optional[float] = None  # Initialize avg_cv_loss attribute

    def log(self, mode):

        # MODIFIED LOGGING FUNCTION TO HANDLE MULTIPLE PARAMETERS

        file_name = f"{self.model_obj.name}_{self.trainer_obj.name}_log.txt"

        with open(file_name, mode) as f:
            if mode == "w":
                f.write("### Log file ###\n\n")
                f.write(f"ML model used: {self.model_obj.name}\n\n")
                f.write("Initial Parameters:\n")

                # NEW: Log SVR specific parameters
                if self.model_obj.name == "support_vector_regression":
                    f.write("Model Parameters (SVR):\n")
                    f.write(f"C: {self.model_obj.svr_params.get('C', 'N/A')}\n")
                    f.write(
                        f"Kernel: {self.model_obj.svr_params.get('kernel', 'N/A')}\n"
                    )
                    f.write(f"Gamma: {self.model_obj.svr_params.get('gamma', 'N/A')}\n")
                    f.write("\n")

                if self.model_obj.train_parameters:
                    # Initial Parameters for gradient based models
                    for i, param in enumerate(self.model_obj.train_parameters):
                        # write each parameter (weights, bias, etc.) on a new line
                        f.write(f"Parameter {i} (shape: {list(param.shape)}):\n")
                        f.write(f"{str(param.tolist())}\n")

            else:
                f.write("\n")

                # SVR logic; no iteration, no weights, no final loss
                if self.model_obj.name == "support_vector_regression":
                    f.write("SVR final results (analytical model\n")
                    f.write(f"Model training; completed during initialization.\n")
                    f.write(f"Iterative training not applicable for SVR.\n")

                # Gradient based models logging
                else:
                    f.write(f"Training algorithm: {self.trainer_obj.name}\n\n")
                    f.write("Hyperparameters:\n")
                    f.write(f"Eta: {str(self.trainer_obj.eta)}\n")
                    f.write("Training results:\n\n")

                    # Final Parameters after training
                    for i, param in enumerate(self.model_obj.train_parameters):
                        # write each parameter (weights, bias, etc.) on a new line
                        f.write(f"Parameter {i} (shape: {list(param.shape)}):\n")
                        f.write(f"{str(param.tolist())}\n")

                    # Final loss for iterative models
                    f.write(
                        f"Loss function value after {str(self.trainer_obj.n_iter)} iterations is : {str(self.trainer_obj.avg_losses[-1])}"
                    )

                f.write("---------------------------------------------------\n")
                # Log the cross validation results if available
                if self.avg_cv_loss is not None:
                    f.write(
                        f"Average Cross-Validation Loss (MSE): {self.avg_cv_loss:.6f}\n"
                    )
                else:
                    f.write("Cross-Validation not performed.\n")

    """
            with open(self.name,mode) as f:
                if mode == 'w':
                    f.write('### Log file ###\n\n')
                    f.write(f'ML model used: {self.model_obj.name}\n\n')
                    f.write('Initial Parameters:\n')
                    f.write(f'Weights: {str(self.model_obj.weights.tolist())}\n')
                    f.write(f'Bias: {str(self.model_obj.bias.tolist())}\n')

                else:
                    f.write('\n')
                    f.write(f'Training algorithm: {self.trainer_obj.name}\n\n')
                    f.write('Hyperparameters:\n')
                    f.write(f'Eta: {str(self.trainer_obj.eta)}\n')
                    f.write('Training results:\n\n')
                    f.write(f'Weights: {str(self.model_obj.weights.tolist())}\n')
                    f.write(f'Bias: {str(self.model_obj.bias.tolist())}\n')
                    f.write(f'Loss function value after {str(self.trainer_obj.n_iter)} iterations is : {str(self.trainer_obj.losses[-1])}')
                """
