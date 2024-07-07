import numpy as np

def validation_training_split(inner_loop_splits):
    folds = []
    for k in range(len(inner_loop_splits)):
        validation_indices = inner_loop_splits[k]
        train_indices = np.concatenate(inner_loop_splits[:k] + inner_loop_splits[k+1:])
        folds.append([train_indices, validation_indices])
    return folds
