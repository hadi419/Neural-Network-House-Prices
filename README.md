# Neural Network Model, Machine Learning

## Part 1

The neural network mini-library is found part1_nn_lib.py. When running this file with the python3 part1_nn_lib.py command, a simple test on the library gets executed.

## Part 2 - Create and train a neural network for regression

### Regressor Class

This is found in part2_house_value_regression.py.

### Hyperparameter Search

To test it, we have three different functioning hyperparameter search functions in:
- deprec_hyperparam_part2.py: manually implemented grid hyperparameter search with for loops)
- random_hyperparam_part2.py: Random search across hyperparameter space for tuning
- half_grid_hyperparam_part2.py: Tournament based halving grid search across hyperparameter space. This proves to be more efficient than normal grid search.

There is also bayesian_hyperparam_part2.py which implements bayesian-based hyperparameter tuning, however this faces several issues with deprecated libraries.

Run them with a python3 command, and after training with cross-validation, the overall score against the held-out test set is printed, and the regressor is automatically stored in part2_model.pickle. For our final commit, we've only kept best_part2_model.pickle, and the load function has been appropriately changed.
