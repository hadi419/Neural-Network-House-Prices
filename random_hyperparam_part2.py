from numpy.random import default_rng
import pandas as pd
from part2_house_value_regression import *

from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import loguniform, uniform, randint # import other distributions for the hyperparameter space

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

    regressor = Regressor(x, hidden_size=7, num_hidden=3, nb_epoch=500, lr=0.01, weight_decay=0.0001)
    
    # Randomised based Hyperparameter tuning:
    # Define hyperparameter space:
    # The keys in the dictionary must match the hyperparameter variables given as input to regressor constructor:
    param_space = {
#            'hidden_size': randint(2, 8),
#            'num_hidden': randint(3, 20),
#            'nb_epoch': ,
            'lr': loguniform(1e-5, 1e-1),
#            'weight_decay': loguniform(1e-6, 1e-3)
        }

    random_search = RandomizedSearchCV(regressor, 
                                       param_distributions=param_space, 
                                       n_iter=5, 
                                       cv=KFold(n_splits=5, shuffle=False))

    random_search.fit(x_train, y_train)

    best_param = random_search.best_params_
    best_regressor = random_search.best_estimator_
    print("best_Score", random_search.best_score_)
    print("Best parameters: ", best_param)
    # print("Best regressor: ", best_regressor)


    # Evaluate accuracy on held-out test set:
    ### FUTURE: Ideally, we would retrain the regressor on all the non-test data with the new hyperparameter values.
    ### We should also retrain the regressor on all of the train data to give as a final most-efficient neural network.
    x_test = x.iloc[test_indices, :]
    y_test = y.iloc[test_indices, :]

    print("")
    test_error = random_search.score(x_test, y_test)
    print("Overall performance: ", test_error)

    # return  # Return the chosen hyper parameters
    return best_regressor


def hyperparameter_main():
    data = pd.read_csv("housing.csv") 
    regressor = RegressorHyperParameterSearch(data)
    save_regressor(regressor)

    ### Future: Get the best_parameters, train on entire dataset and submit this as pickle.


if __name__ == "__main__":
    hyperparameter_main()

