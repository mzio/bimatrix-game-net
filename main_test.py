import numpy as np
import pandas as pd
import argparse
import itertools

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from data_utils import get_payoff_matrices, transform, expand_channels, get_splits
from dataloader import load_game_data
from train import evaluate
from model import ConvNetBig

if __name__ == "__main__":

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Cuda
    DEVICE = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load test data
    df_test_features = pd.read_csv('./hb_test_feature.csv')

    # Prepare data for dataloader
    norm_payoffs_test = get_payoff_matrices(
        df_test_features, rows_first=True, normalized=True)
    norm_payoffs_test_r = norm_payoffs_test[:200]
    norm_payoffs_test_c = norm_payoffs_test[200:]

    augmented_data_test = []
    orderings = list(itertools.permutations([0, 1, 2]))
    for order in orderings:
        trans_payoffs_test_r = transform(norm_payoffs_test_r, order)
        trans_payoffs_test_c = transform(norm_payoffs_test_c, order)
        augmented_data_test.append(
            np.array([trans_payoffs_test_r, trans_payoffs_test_c]))

    game_data_test = np.array(augmented_data_test).transpose(
        [0, 2, 1, 3, 4]).reshape(200 * 6, 2, 3, 3)

    game_data_test_rc_diffs = (game_data_test[:, 0, :, :] -
                               game_data_test[:, 1, :, :])

    max_rows = np.max(game_data_test[:, 0, :, :], axis=(1, 2))
    max_cols = np.max(game_data_test[:, 1, :, :], axis=(1, 2))

    # Expand and convert to previous data shape
    max_rows = np.repeat(max_rows, 9).reshape((1200, 3, 3))
    max_cols = np.repeat(max_cols, 9).reshape((1200, 3, 3))

    game_data_test_maxdiff_r = game_data_test[:, 0, :, :] - max_rows
    game_data_test_maxdiff_c = game_data_test[:, 1, :, :] - max_cols

    game_data_test_diffs = np.array([game_data_test_rc_diffs,
                                     game_data_test_maxdiff_r,
                                     game_data_test_maxdiff_c]).transpose(1, 0, 2, 3)
    game_data_test = np.concatenate([game_data_test,
                                     game_data_test_diffs], axis=1)

    # Setup dataloader
    labels = np.zeros([game_data_test.shape[0], 3])
    dataloader_test = load_game_data(game_data_test[:200], labels, batch_size=1,
                                     shuffle=False, seed=42, device=DEVICE)

    # Load models
    model_paths = ['./models/model_final_all_mse_e=200_s=42_do=0.5.pt',
                   './models/model_final_all_mse_e=400_s=42_do=0.5.pt',
                   './models/model_final_all_mse_e=500_s=42_do=0.5.pt',
                   './models/model_final_all_mse_e=600_s=42_do=0.5.pt']

    # Calculate predictions for each model
    print('Calculating model predictions...')
    dfs = []
    for model_path in tqdm(model_paths):
        model_final = ConvNetBig(
            log_softmax=False, softmax=True, dropout=0.5, channels=5)
        if str(DEVICE) == 'cpu':
            model_final.load_state_dict(torch.load(model_path,
                                                   map_location='cpu'))
        else:
            model_final.load_state_dict(torch.load(model_path))
        model_final.to(DEVICE)
        model_final.eval()

        batch_ql, acc, predictions = evaluate(model_final, dataloader_test)
        pred_actions = np.argmax(np.array(predictions), axis=1)
        predictions = np.array(predictions).squeeze()
        df = pd.DataFrame(predictions, columns=['f1', 'f2', 'f3'])
        df['action'] = pred_actions
        dfs.append(df)

    # Average ensembled predictions
    avg_freq = np.array([
        np.mean([dfs[i]['f1'] for i in range(4)], axis=0),
        np.mean([dfs[i]['f2'] for i in range(4)], axis=0),
        np.mean([dfs[i]['f3'] for i in range(4)], axis=0)
    ]).transpose([1, 0])

    actions = np.argmax(np.array(avg_freq), axis=1) + 1.

    df_avg = pd.DataFrame(avg_freq, columns=['f1', 'f2', 'f3'])
    df_avg['action'] = actions

    df_avg.to_csv('hb_test_pred.csv', index=False)
    print('Data saved to ./hb_test_pred.csv!')
