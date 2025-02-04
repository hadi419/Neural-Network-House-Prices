import numpy as np
import pickle


def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative
    log-likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Sigmoid layer.
        """
        self._cache_current = None

    @staticmethod
    def sigmoid(x):
        """
        Apply sigmoid function element-wise to a NumPy array.

        Arguments:
            x {np.ndarray} -- Input array.

        Returns:
            {np.ndarray} -- Result of applying sigmoid function element-wise on the input.
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        """
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        # Caches result of forward pass through the sigmoid function
        self._cache_current = SigmoidLayer.sigmoid(x)
        return self._cache_current

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        return grad_z * (self._cache_current * (1 - self._cache_current))


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        self._cache_current = None

    @staticmethod
    def relu(x):
        """
        Apply ReLU activation function element-wise to a NumPy array.

        Arguments:
            x {np.ndarray} -- Input array.

        Returns:
            {np.ndarray} -- Result of applying ReLU activation function element-wise.
        """
        return np.maximum(0, x)

    def forward(self, x):
        """
        Performs forward pass through the Relu layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        # Caches result of forward pass through the ReLU function
        self._cache_current = ReluLayer.relu(x)
        return self._cache_current

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        return grad_z * np.where(self._cache_current > 0, 1, 0)


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in  = n_in
        self.n_out = n_out

        # Initializes W using the Xavier/Glorot function
        self._W = xavier_init((n_in, n_out))
        self._b = np.zeros((1, n_out))

        self._cache_current  = None
        self._grad_W_current = None
        self._grad_b_current = None

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        # Ensures matrix shapes match before any calculations
        assert x.shape[1] == self._W.shape[0]
        assert self._W.shape[1] == self._b.shape[1]

        # Caches the transpose of X
        # Used for calculating the gradient in the backwards propagation
        self._cache_current = x.T

        # @ used for matrix multiplication
        # Broadcasts self._b to shape of x @ self._W
        return (x @ self._W) + self._b

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        # Ensures matrix shapes match before any calculations
        assert self._cache_current.shape[1] == grad_z.shape[0]
        assert grad_z.shape[1] == self._W.T.shape[0]

        # Calculates and caches gradients of Loss wrt b and W
        # @ used for matrix multiplication
        self._grad_b_current = np.ones((1, grad_z.shape[0])) @ grad_z
        self._grad_W_current = self._cache_current @ grad_z

        # Returns gradient of Loss wrt x (the input of the layer)
        return grad_z @ self._W.T

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        # Updates W and b parameters by applying one step of gradient descent
        self._W = self._W - (learning_rate * self._grad_W_current)
        self._b = self._b - (learning_rate * self._grad_b_current)


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding
                the batch dimension).
            - neurons {list} -- Number of neurons in each linear layer
                represented as a list. The length of the list determines the
                number of linear layers.
            - activations {list} -- List of the activation functions to apply
                to the output of each linear layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations
        self._layers = []

        activation_mappings = {
            "sigmoid": SigmoidLayer,
            "relu": ReluLayer,
        }

        # Iterates over the list of activation functions for each layer
        for i, activation in enumerate(activations):
            # Always appends a linear layer
            self._layers.append(LinearLayer(input_dim, neurons[i]))

            # Appends relevant activation layer if given
            # Raises error if activation function is not included
            if activation in activation_mappings:
                self._layers.append(activation_mappings[activation]())
            elif activation != "identity":
                raise ValueError(f'Unknown activation function: {activation}')

            # Updates the input dimension for the next layer with the output dimension of the current layer
            input_dim = neurons[i]

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        # Iteratively computes forward propagation for each layer in the network
        for layer in self._layers:
            x = layer.forward(x)

        # Ensures the output layer matrix shape aligns with the defined number of output neurons before returning it
        assert x.shape[1] == self.neurons[-1]
        return x

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, input_dim).
        """
        # Iteratively computes backwards propagation for each layer in the network
        for layer in reversed(self._layers):
            grad_z = layer.backward(grad_z)

        # Ensures the shape of the gradient of Loss wrt the input matches the input dimensions before returning it
        assert grad_z.shape[1] == self.input_dim
        return grad_z

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        # Updates the parameters of each layer and passes if it is not a linear layer
        for layer in self._layers:
            layer.update_params(learning_rate)


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                cross_entropy.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        loss_mappings = {
            "cross_entropy": CrossEntropyLossLayer,
            "mse": MSELossLayer,
        }

        # Appends relevant loss layer if given
        # Raises error if loss function is not included
        if loss_fun in loss_mappings:
            self._loss_layer = loss_mappings[loss_fun]()
        else:
            raise ValueError(f'Unknown loss function: {loss_fun}')

    @ staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns:
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """
        # Ensures that the input and target matrices have the same shape
        assert input_dataset.shape[0] == target_dataset.shape[0]

        # Shuffles the input and target matrices in the same pattern
        shuffled_indices = np.random.permutation(input_dataset.shape[0])
        shuffled_inputs  = input_dataset[shuffled_indices]
        shuffled_targets = target_dataset[shuffled_indices]
        return shuffled_inputs, shuffled_targets

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """
        # Ensures the input and target matrices have the same shape
        assert input_dataset.shape[0] == target_dataset.shape[0]
        dataset_size = input_dataset.shape[0]

        dataset_size = input_dataset.shape[0]
        # Iterates over the whole dataset on multiple epochs
        for epoch in range(self.nb_epoch):

            # Shuffles the data if the shuffle flag is set
            if self.shuffle_flag:
                input_dataset, target_dataset = Trainer.shuffle(input_dataset, target_dataset)

            # Splits the datasets into batches
            n_batches = int(np.ceil(dataset_size / self.batch_size))
            input_batches = np.array_split(input_dataset, n_batches)
            target_batches = np.array_split(target_dataset, n_batches)

            # Iterates over each batch, predicts the output, performs forward & backward propagation & updates parameters using gradient descent
            for i in range(n_batches):
                prediction_batch = self.network(input_batches[i])
                self._loss_layer(prediction_batch, target_batches[i])
                self.network.backward(self._loss_layer.backward())
                self.network.update_params(self.learning_rate)

    def eval_loss(self, input_dataset, target_dataset):
        """

        Function that evaluate the loss function for given data. Returns
        scalar value.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).

        Returns:
            a scalar value -- the loss
        """
        # Calculates loss based on the predicted output dataset
        prediction_dataset = self.network(input_dataset)
        return self._loss_layer(prediction_dataset, target_dataset)


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """
        self._min = np.min(data, axis=0)
        self._max = np.max(data, axis=0)

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        # Normalizes the data using min-max scaling
        return (data - self._min) / (self._max - self._min)

    def revert(self, data):
        """
        Revert the pre-processing operations to retrieve the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        # Reverts previous normalization of the data
        return (data * (self._max - self._min)) + self._min


def example_main():
    input_dim = 4
    neurons = [16,3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()
