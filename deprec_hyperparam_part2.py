from validation_split import validation_training_split
from numpy.random import default_rng
import pandas as pd
from part2_house_value_regression import *


def RegressorHyperParameterSearch(data, k=10):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        

    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    """
    Only have one of the folds as a held-out test set.
    Iterate k-1 times using cross-validation to tune hyperparameters appropriately.
    Each iteration would build different neural networks with different hyperparameters each.
    """
    output_label = "median_house_value"
    # Separate input from output:
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    shuffled_indices = default_rng().permutation(len(data))
    splits = np.array_split(shuffled_indices, k)
    # The last split will be used as the held-out test set. The choice is arbitrary
    test_indices = splits[-1]

    best_error = float('inf')
    for (train_indices, validation_indices) in validation_training_split(splits[:-1]):
        x_train = x.iloc[train_indices, :]
        y_train = y.iloc[train_indices, :]
        x_validation = x.iloc[validation_indices, :]
        y_validation = y.iloc[validation_indices, :]

        # Hyperparameter tuning:
        ### FUTURE: Implement more complex methods for hyperparameter tuning, e.g. grid search, random search
        ### or more advanced optimisation algorithms for the hyperparameter space.
        for neuron_num in [5, 10]:
            # Build and evaluate each neural network
            regressor = Regressor(x_train, num_hidden=neuron_num, hidden_size=8, min_delta=0, patience=10, nb_epoch=5000)
            regressor.optimiser = torch.optim.Adam(regressor.net.parameters(), lr=0.0008, weight_decay=0)
            # weight_decay > 0 implements L2 regularisation.
            regressor.fit(x_train, y_train, x_validation, y_validation)

            error = regressor.score(x_validation, y_validation)
            print("Regressor fitted for neuron_num=", neuron_num, ", with error: ", error)
            # print("error, ", error)
            # Compare performance to previous:
            if (error < best_error):
                best_error = error
                print("New best validation error:", best_error)
                best_regressor = regressor
                # save_regressor(best_regressor)

    # Evaluate accuracy on held-out test set:
    ### FUTURE: Ideally, we would retrain the regressor on all the non-test data with the new hyperparameter values.
    x_test = x.iloc[test_indices, :]
    y_test = y.iloc[test_indices, :]

    test_error = regressor.score(x_test, y_test)
    print("\nOverall performance: ", test_error)

    # return  # Return the chosen hyper parameters
    return best_regressor

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def hyperparameter_main():
    data = pd.read_csv("housing.csv") 
    regressor = RegressorHyperParameterSearch(data)

    save_regressor(regressor)


if __name__ == "__main__":
    hyperparameter_main()


# def example_main():

#     output_label = "median_house_value"

#     # Use pandas to read CSV data as it contains various object types
#     # Feel free to use another CSV reader tool
#     # But remember that LabTS tests take Pandas DataFrame as inputs
#     data = pd.read_csv("housing.csv") 

#     # Splitting input and output
#     x_train = data.loc[:, data.columns != output_label]
#     y_train = data.loc[:, [output_label]]

#     # Training
#     # This example trains on the whole available dataset. 
#     # You probably want to separate some held-out data 
#     # to make sure the model isn't overfitting
#     # regressor = Regressor(x_train, nb_epoch=1000, dropout=0.2)
#     # regressor.optimiser = torch.optim.Adam(regressor.net.parameters(), lr=0.01, weight_decay=0.00001) # Use custom optimsier
#     # For L2 regularisation, set weight_decay > 0.
#     # regressor.fit(x_train, y_train)
#     # save_regressor(regressor)

#     # Load Regressor
#     # regressor = load_regressor

#     # Error
#     error = regressor.score(x_train, y_train)
#     print("\nRegressor error: {}\n".format(error))


# if __name__ == "__main__":
#     example_main()

