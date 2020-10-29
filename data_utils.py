import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def view_game_matrix(df_row):
    """
    View payoffs in 3x3 matrix format
    args:
        df_row : Pandas series object, e.g. df.iloc[0]
    returns
        [row_payoffs, col_payoffs] : np.array (2x3x3)
    """
    return df_row.values.reshape(2, 3, 3)


def normalize(matrix):
    """
    Method to normalize a given matrix
    args:
        matrix : np.array, the payouts 
    output:
        np.array, the matrix normalized
    """
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)


def get_payoff_matrices(df, rows_first=False, normalized=True):
    """
    Convert input dataframe into new normed payoff data
    args:
        df : pandas.DataFrame
        rows_first : bool, if true, separate data into first row payoffs and then columns, 
        otherwise row and col payoffs lumped next to each other
        normalized : bool, if true, normalize the matrices
    output:
        np.array, the converted data 
    """
    if normalized:
        df = df.apply(normalize, axis=0)
    matrices = np.zeros([2 * df.shape[0], 3, 3])
    for row_ix in range(df.shape[0]):
        payoffs = view_game_matrix(df.iloc[row_ix])
        if rows_first:
            matrices[row_ix] = payoffs[0]
            matrices[row_ix + df.shape[0]] = payoffs[1]
        else:
            matrices[row_ix * 2] = payoffs[0]
            matrices[row_ix * 2 + 1] = payoffs[1]
    return matrices


def transform(payoffs, rotations):  # rotations = [2, 1, 0]
    new_payoffs = np.zeros(payoffs.shape)
    for ix in range(len(rotations)):
        new_payoffs[:, ix] = payoffs[:, rotations[ix]]
    return new_payoffs


def expand_channels(game_data, channels='all'):
    """
    Return alternate channels for model input
    args:
        channels : str, one of 'all', 'payoffs', 'diffs', 'row'
    """
    if channels == 'payoffs':
        return game_data
    # Difference between the row and col payoffs
    game_data_rc_diffs = game_data[:, 0, :, :] - game_data[:, 1, :, :]

    # Difference between the payoff max and payoffs
    # Get maxes
    max_rows = np.max(game_data[:, 0, :, :], axis=(1, 2))
    max_cols = np.max(game_data[:, 1, :, :], axis=(1, 2))

    # Expand and convert to previous data shape
    max_rows = np.repeat(max_rows, 9).reshape((1500, 3, 3))
    max_cols = np.repeat(max_cols, 9).reshape((1500, 3, 3))

    game_data_maxdiff_r = game_data[:, 0, :, :] - max_rows
    game_data_maxdiff_c = game_data[:, 1, :, :] - max_cols

    game_data_diffs = np.array(
        [game_data_rc_diffs, game_data_maxdiff_r, game_data_maxdiff_c]).transpose(1, 0, 2, 3)
    game_data_combined = np.concatenate([game_data, game_data_diffs], axis=1)
    game_data_row = np.concatenate([np.expand_dims(
        game_data[:, 0, :, :], 1), game_data_diffs[:, 0:2, :, :]], axis=1)

    if channels == 'diffs':
        return game_data_diffs
    elif channels == 'row':
        return game_data_row
    else:
        return game_data_combined


def get_splits(data, labels, num_splits):
    """
    Returns splits for data
    Output:
        Array of splits, each split = [train_data, train_labels, 
                                       test_data, test_labels]
    """
    splits = []
    kf = KFold(n_splits=num_splits)
    for train_val_ix, test_ix in kf.split(data):
        train_val_data, test_data = np.array(
            data[train_val_ix]), np.array(data[test_ix])
        train_val_labels, test_labels = np.array(
            labels[train_val_ix]), np.array(labels[test_ix])
        splits.append(
            [train_val_data, train_val_labels, test_data, test_labels])
    return splits
