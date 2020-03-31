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

    for order in orderings:
        trans_payoffs_test_r = transform(norm_payoffs_test_r, order)
        trans_payoffs_test_c = transform(norm_payoffs_test_c, order)
        augmented_data_test.append(
            np.array([trans_payoffs_test_r, trans_payoffs_test_c]))

    game_data_test = np.array(augmented_data_test).transpose(
        [0, 2, 1, 3, 4]).reshape(200 * 6, 2, 3, 3)

    game_data_test = expand_channels(game_data_test, channels=args.channel)

    # Setup dataloader
    labels = np.zeros([game_data_test.shape[0], 3])
    dataloader_test = load_game_data(
        game_data_test[:200], labels, batch_size=1, shuffle=False)

    # Load models
    model_paths = ['./models/model_final_all_mse_e=200_s=42_do=0.5.pt',
                   './models/model_final_all_mse_e=400_s=42_do=0.5.pt',
                   './models/model_final_all_mse_e=500_s=42_do=0.5.pt',
                   './models/model_final_all_mse_e=600_s=42_do=0.5.pt']

    # Calculate predictions for each model
    dfs = []
    for model_path in model_paths:
        model_final = ConvNetPdBig(
            log_softmax=False, softmax=True, dropout=0.5, channels=5)
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
    averages = np.array([
        np.mean([dfs[i]['f1'] for i in range(4)], axis=0),
        np.mean([dfs[i]['f2'] for i in range(4)], axis=0),
        np.mean([dfs[i]['f3'] for i in range(4)], axis=0)
    ]).transpose([1, 0])

    pd.DataFrame(averages, columns=['f1', 'f2', 'f3']).to_csv(
        'hb_model_final_avg_test.csv', index=False)
