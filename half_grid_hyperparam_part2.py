### This performs grid search with halving, as supported by the scikit-learn package.
### It is a tournament based grid search method where each additional iteration,
### fewer neural networks get through and more samples from the data is fed to the neural networks that passed.

from numpy.random import default_rng
import pandas as pd
from part2_house_value_regression import *
from sklearn.experimental import enable_halving_search_cv  # noqa # This is needed to confirm functionality of halvinggridsearch, an experimental scikit feature.

from sklearn.model_selection import HalvingGridSearchCV, KFold


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

    regressor = Regressor(x, hidden_size=7, num_hidden=3, nb_epoch=300, lr=0.01, weight_decay=0.0001)
    
    # Define hyperparameter space:
    # param_space = {'hidden_size': np.arange(16, 128, 16), 'lr': np.logspace(-5, -1, 5)} # These are just lists.
    # The keys in the dictionary must match the hyperparameter variables given as input to regressor constructor.
    param_space = {
#            'hidden_size': ,
#            'num_hidden': ,
#            'nb_epoch': ,
            'lr': [1e-3, 1e-2, 1e-1], # Can change to np.logspace
#            'weight_decay': [0, 1e-5, 1e-4, 1e-3, 1e-2]
        }

    halving_grid_search = HalvingGridSearchCV(regressor, 
                                       param_grid=param_space,
                                       factor=3,
                                       resource='n_samples',
                                       min_resources='exhaust', # 183 as 14850(number of training samples) / 3^4
                                       aggressive_elimination=False, # True to eliminate more candidates before final iteration
                                       cv=5)
    halving_grid_search.fit(x_train, y_train)

    best_param = halving_grid_search.best_params_
    best_regressor = halving_grid_search.best_estimator_
    print("best_Score", halving_grid_search.best_score_)
    print("Best parameters: ", best_param)
    print("Best regressor: ", best_regressor)


    # Evaluate accuracy on held-out test set:
    ### FUTURE: Ideally, we would retrain the regressor on all the non-test data with the new hyperparameter values.
    x_test = x.iloc[test_indices, :]
    y_test = y.iloc[test_indices, :]

    test_error = halving_grid_search.score(x_test, y_test)
    print("\nOverall performance: ", test_error)

    # return  # Return the chosen hyper parameters
    return best_regressor


def hyperparameter_main():
    data = pd.read_csv("housing.csv") 
    regressor = RegressorHyperParameterSearch(data)

    save_regressor(regressor)


if __name__ == "__main__":
    hyperparameter_main()