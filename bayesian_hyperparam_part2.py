### This was an attempt at Bayesian hyperparameter optimisation, but there are many issues with the skopt library that implements it.
### Although it should've simply been a change from randomsearchcv to bayessearchcv, I have found multiple mistakes in the library
### and compatibility issues with newer versions of scikit-learn libraries, so rolling back to accommodate for these would then make
### the code further incompatible with LabTS tests.

from numpy.random import default_rng
import pandas as pd
from part2_house_value_regression import *

from sklearn.model_selection import KFold
from skopt import BayesSearchCV
# from sklearn.metrics import make_scorer
# from scipy.stats import randint, uniform # import other distributions for the hyperparameter space

### I don't believe this type of function is necessary
# def optimise_params(learning_rate, weight_decay):
#     return #negative of score, so that it is maximised to as positive as possible

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
    
    output_label = "median_house_value"
    # Separate input from output:
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    shuffled_indices = default_rng().permutation(len(data))
    splits = np.array_split(shuffled_indices, k)
    # The last split will be used as the held-out test set. The choice is arbitrary
    test_indices = splits[-1]
    train_indices = np.concatenate(splits[:-1])


    x_train = x.iloc[train_indices, :]
    y_train = y.iloc[train_indices, :]

    regressor = Regressor(x, num_hidden=10, hidden_size=5, min_delta=0, patience=10, nb_epoch=250)
    
    # Randomised based Hyperparameter tuning:
    # Define hyperparameter space:
    ### FUTURE: Give parameters as a continuous distribution using scipy.stats distributions. Logarithmic distributions?
    # param_space = {'hidden_size': np.arange(16, 128, 16), 'learning_rate': np.logspace(-5, -1, 5)} # These are just lists.
    # The keys in the dictionary must match the hyperparameter variables given as input to regressor constructor.
    param_space = {'lr': [1e-5, 1e-4, 1e-3, 1e-2], 'weight_decay': [0, 1e-5, 1e-4, 1e-3, 1e-2]}

    # scorer = make_scorer()

    random_search = BayesSearchCV(regressor, 
                                       search_spaces=param_space, 
                                       n_iter=5, 
                                       cv=KFold(n_splits=5, shuffle=False))
    print("Before .fit()")
    random_search.fit(x_train, y_train)

    best_param = random_search.best_params_
    best_regressor = random_search.best_estimator_
    print("best_Score", random_search.best_score_)
    print("Best parameters: ", best_param)
    print("Best regressor: ", best_regressor)


    # Evaluate accuracy on held-out test set:
    ### FUTURE: Ideally, we would retrain the regressor on all the non-test data with the new hyperparameter values.
    x_test = x.iloc[test_indices, :]
    y_test = y.iloc[test_indices, :]

    test_error = random_search.score(x_test, y_test)
    print("\nOverall performance: ", test_error)

    # return  # Return the chosen hyper parameters
    return best_regressor


def hyperparameter_main():
    data = pd.read_csv("housing.csv") 
    regressor = RegressorHyperParameterSearch(data)

    # save_regressor(regressor)


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

