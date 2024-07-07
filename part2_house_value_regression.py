import torch
import torch.nn as nn
# import torch.nn.functional as F
from sklearn.preprocessing import LabelBinarizer
import pickle
import numpy as np

from net import *
from early_stopper import *

class Regressor():

    def __init__(self, x, hidden_size=7, num_hidden=3, activation='ReLU', dropout=0, nb_epoch=1000, 
                 min_delta=0, patience=1, lr=0.005, weight_decay=0.0001):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - hidden_size {int} -- number of neurons for each hidden layer.
            - num_hidden {int} -- the number of hidden layers.
            - activation {str} -- the type of activation function for hidden layers. Can be one of the following:
                ReLU, Sigmoid, Tanh.
            - criterion {torch.nn.?} -- the loss function to use.
            - dropout {float} -- what probability to drop neurons with during training. If zero, no dropout.
            - nb_epoch {int} -- number of epochs to train the network.
            - min_delta {int} -- a value to add to the minimum validation loss so far, which gives the early stopping threshold.
            - patience {int} -- how many epochs we can tolerate the validation loss being above the early stopping threshold
                before we stop.
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Preprocessing.
        X, _ = self._preprocessor(x, training=True)

        # Fixed parameters.
        self.x = x
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden
        self.activation = activation ### Future: implement different activation functions
        self.dropout = dropout
        self.nb_epoch = nb_epoch
        self.min_delta = min_delta
        self.patience = patience
        self.lr = lr 
        self.weight_decay = weight_decay

        # Other values
        input_size = X.shape[1]
        output_size = 1

        # Create network.
        self.net = Net(input_size, self.hidden_size, output_size, self.num_hidden, self.activation, self.dropout)
        if torch.cuda.is_available():
            self.net.to('cuda')

        
        # Default optimiser.
        # self.optimiser = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        # return x, (y if isinstance(y, pd.DataFrame) else None)
        
        # Handle possible missing attributes.
        # Note that the only column with missing attributes is number of bedrooms.
        if training: # If training, find mean value
            self.mean_bedrooms = x['total_bedrooms'].mean()
        X = x.fillna(self.mean_bedrooms)

        # Convert from Pandas dataframe to Numpy array (easier for label binariser)
        X = X.to_numpy()
        Y = None
        if y is not None:
            Y = y.to_numpy(dtype=np.float64, copy=True)

        # The only attribute with textual values is ocean proximity.
        # Possible values: INLAND, <1H OCEAN, NEAR BAY, NEAR OCEAN, ISLAND
        if training: # If training, initialise binariser
            self.lb = LabelBinarizer()
            # self.lb.fit(X[:, 8]) # Find labels
            self.lb.fit(np.array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])) # Hard-code labels
        one_hot_op = self.lb.transform(X[:, 8]) # Convert labels to one-hot encoding
        X = X[:, :8] # Remove labels
        X = np.concatenate((X, one_hot_op), axis=1) # Add one-hot encoding

        # Normalise x and y.
        X = X.astype(np.float64) # Force type to float AFTER switching to one-hot encoding
        if training: # If training, find normalisation factors
            self.first_col_norm = X[:, 0].min(axis=0) # First column is always negative
            self.other_cols_norm = X[:, 1:8].max(axis=0)
        X[:, 0] /= self.first_col_norm
        X[:, 1:8] /= self.other_cols_norm
        if Y is not None:
            if training:
                self.y_norm = Y.max(axis=0)[0]
            Y /= self.y_norm

        # Convert from Numpy array to Torch tensor.
        x_tensor = torch.from_numpy(X).float()
        x_tensor.requires_grad = True
        if torch.cuda.is_available():
            x_tensor = x_tensor.to('cuda')
            
        if Y is not None:
            y_tensor = torch.from_numpy(Y).float()
            if torch.cuda.is_available():
                y_tensor = y_tensor.to('cuda')
            return x_tensor, y_tensor
        else:
            return x_tensor, None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x_train, y_train, x_val=None, y_val=None):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Print statements to follow changes in hyperparameters during cross-validation
        print("testing with lr:", self.lr)
        # print("testing with nb_epoch:", self.nb_epoch)
        # print("testing with hidden_size:", self.hidden_size)
        # print("testing with num_hidden:", self.num_hidden)
        print("testing with weight_decay:", self.weight_decay)

        has_val = x_val is not None and y_val is not None

        X_train, Y_train = self._preprocessor(x_train, y=y_train, training=True) # Do not forget
        if has_val:
            X_val, Y_val = self._preprocessor(x_val, y=y_val, training=False)
        self.net.train() # Set model to training mode

        # Create early stopper.
        early_stopper = EarlyStopper(self.patience, self.min_delta)
        early_stopper.reset()

        criterion = nn.MSELoss()
        
        for epoch in range(self.nb_epoch):
            self.optimiser.zero_grad() # Reset gradients
            y_hat = self.net(X_train) # Forward pass
            train_loss = criterion(y_hat, Y_train) # Compute loss
            if epoch % 10 == 0:
                train_loss_norm = torch.mul(torch.sqrt(train_loss), self.y_norm)
                # print("train_loss unnormalised: ", epoch, train_loss_norm)
            train_loss.backward() # Backward pass
            self.optimiser.step() # Update parameters
            if has_val:
                y_val_hat = self.net(X_val)
                val_loss = criterion(y_val_hat, Y_val)
                print("validation loss:", epoch, train_loss, val_loss)
                if early_stopper.stop(val_loss):
                    print("Stopped at epoch:", epoch)
                    break
        
        print("final train_loss_norm:", epoch, train_loss_norm)

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        self.net.eval() # Set model to evaluation mode

        y_hat = self.net(X) # The same as forward passing
        y_hat = y_hat.detach().cpu().numpy() # Changes y_hat into a np.ndarray type.
        y_hat *= self.y_norm
        
        return y_hat

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        self.net.eval() # Set model to evaluation mode

        y_hat = self.net(X) # The same as forward passing
        evaluate = nn.MSELoss() #For MSE loss.
        error = torch.sqrt(evaluate(y_hat, Y)) # Compute RMSE loss.
        error = torch.mul(error, self.y_norm)
        print("error from score method:", error)
        return -error # It's negative so that maximising the output (performance) is equivalent to a smaller sized error.

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    
    # To replicate the scikit-learn estimator.
    # Must match the constructor arguments. From scikit-learn user guide.
    def get_params(self, deep=True):
        return {
            'x': self.x,
            'hidden_size': self.hidden_size,
            'num_hidden': self.num_hidden,
            'activation': self.activation,
            'dropout': self.dropout,
            'nb_epoch': self.nb_epoch,
            'min_delta': self.min_delta,
            'patience': self.patience,
            'lr': self.lr,
            'weight_decay': self.weight_decay
        }
    
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    

def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('best_part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model

# The different RegressorHyperParameterSearch functions are in *_hyperparam_part2.py files.