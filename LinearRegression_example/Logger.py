
class Logger():
    
    def __init__(self,model_obj,trainer_obj):

        self.name = 'log.txt'
        self.model_obj = model_obj
        self.trainer_obj = trainer_obj
    
    def log(self,mode):
        '''
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
            '''
        
        # MODIFIED LOGGING FUNCTION TO HANDLE MULTIPLE PARAMETERS

        file_name = f'{self.model_obj.name}_{self.trainer_obj.name}_log.txt'

        with open(file_name,mode) as f:
            if mode == 'w':
                f.write('### Log file ###\n\n')
                f.write(f'ML model used: {self.model_obj.name}\n\n')
                f.write('Initial Parameters:\n')

                # Initial Parameters, 
                for i, param in enumerate(self.model_obj.train_parameters):
                    # write each parameter (weights, bias, etc.) on a new line
                    f.write(f'Parameter {i} (shape: {list(param.shape)}):\n')
                    f.write(f'{str(param.tolist())}\n')

            else:
                f.write('\n')
                f.write(f'Training algorithm: {self.trainer_obj.name}\n\n')
                f.write('Hyperparameters:\n')
                f.write(f'Eta: {str(self.trainer_obj.eta)}\n')
                f.write('Training results:\n\n')

                # Final Parameters after training
                for i, param in enumerate(self.model_obj.train_parameters):
                    # write each parameter (weights, bias, etc.) on a new line
                    f.write(f'Parameter {i} (shape: {list(param.shape)}):\n')
                    f.write(f'{str(param.tolist())}\n')

                f.write(f'Loss function value after {str(self.trainer_obj.n_iter)} iterations is : {str(self.trainer_obj.losses[-1])}')