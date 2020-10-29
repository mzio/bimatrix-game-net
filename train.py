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


def batch_quadratic_loss(batch_output, batch_label):
    """
    Calculate quadratic loss metric, i.e. sum over all (pred - label)^2
    """
    loss = 0
    for ix in range(len(batch_output)):
        loss += np.sum([(batch_output[ix][i] - batch_label[ix]
                         [i]) ** 2. for i in range(3)])
    return loss


def evaluate(model, dataloader, model_softmax=False):
    """
    Perform inference on model given dataloader
    Inputs:
        model : PyTorch nn.Module - the trained model
        dataloader : PyTorch Dataloader 
        model_softmax : whether the model performs the softmax operation in its forward pass
    """
    softmax = nn.Softmax(dim=1)
    batch_quad_loss = 0
    outputs = []
    labels = []
    model.eval()
    for batch_ix, (data, label) in enumerate(dataloader):
        if model.soft:
            output = model(data)
        elif model.log:
            # convert back regular probabilities
            output = torch.exp(model(data))
        else:
            output = softmax(model(data))
        batch_quad_loss += batch_quadratic_loss(
            output, label).cpu().detach().numpy()
        outputs.append(output.squeeze().cpu().detach().numpy())
        labels.append(label.squeeze().cpu().detach().numpy())

    val_actions = np.argmax(np.array(labels), axis=1)
    pred_actions = np.argmax(np.array(outputs), axis=1)
    action_acc = np.sum(val_actions == pred_actions) / val_actions.shape[0]

    return batch_quad_loss, action_acc, outputs


def train(model, loss_fn, dataloader, dataloader_val=None, epochs=200,
          log_targets=False, model_softmax=True, log=5, log_softmax_outputs=False,
          val_data=None, val_labels=None, seed=None, device=None):

    softmax_fn = nn.Softmax(dim=1)
    log_softmax_fn = nn.LogSoftmax(dim=1)

    dataloader_actual_val = load_game_data(
        val_data, val_labels, batch_size=1, shuffle=False, device=device, seed=seed)
    loss_total = []
    acc_total = []
    acc_total_val = []
    for epoch in tqdm(range(epochs)):  # tqdm is buggy sometimes
        epoch_loss = 0
        for batch_ix, (data, label) in enumerate(dataloader):
            output = model(data)
            target = torch.tensor(
                np.argmax(np.array(label.cpu()), axis=1), dtype=torch.long).to(device)
            optimizer.zero_grad()
            # loss = 1.0 * mse_loss(softmax(output), label) + 0.0 * ce_loss(output, target)
            if log_targets:
                label = torch.log(label)
            if log_softmax_outputs:
                output = log_softmax_fn(output)
            elif not model_softmax:
                output = softmax_fn(output)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_total.append(epoch_loss)
        quad_loss, acc, _ = evaluate(
            model, dataloader_val, model_softmax=model_softmax)
        quad_loss_val, acc_val, _ = evaluate(
            model, dataloader_actual_val, model_softmax=model_softmax)
        if (epoch + 1) % log == 0:
            print('Epoch {0:<2} | Training Loss: {1:<2f} | Quad Loss: {2:<2f} | Acc: {3:<2f} | Val Quad Loss: {4:<2f} | Val Acc: {5:<2f}'.format(
                epoch, epoch_loss, quad_loss, acc, quad_loss_val, acc_val))
        acc_total.append(acc)
        acc_total_val.append(acc_val)
    return loss_total, [acc_total, acc_total_val]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_path', default='.', type=str)
    parser.add_argument('--save_model_path', default='./models', type=str)
    # Data
    parser.add_argument('--channel', default='all', type=str,
                        help="One of 'all', 'payoffs', 'diffs', 'row'")
    # Model hyperparameters
    parser.add_argument('-bs', '--batch_size', default=4, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('-do', '--dropout', default=0.5, type=float)
    parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float)
    # Training setup
    parser.add_argument('-kf', '--k_folds', default=5, type=str)
    parser.add_argument('--loss', default='mse', type=str,
                        help="Loss criterion. Either 'mse' or 'kl' supported.")
    parser.add_argument('-e', '--epochs', default=200, type=int)
    parser.add_argument('--log', default=50, type=int,
                        help="How frequent to output evaluation metrics during training.")

    args = parser.parse_args()

    args.device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    df_train_features = pd.read_csv(
        '{}/hb_train_feature.csv'.format(args.data_path))
    df_test_features = pd.read_csv(
        '{}/hb_test_feature.csv'.format(args.data_path))
    df_train_truth = pd.read_csv(
        '{}/hb_train_truth.csv'.format(args.data_path))

    # Normalize and convert to 3x3 payoff matrix format
    norm_payoffs = get_payoff_matrices(
        df_train_features, rows_first=True, normalized=True)
    norm_payoffs_r = norm_payoffs[:250]
    norm_payoffs_c = norm_payoffs[250:]

    # Augment data
    orderings = list(itertools.permutations([0, 1, 2]))
    augmented_data = []
    augmented_labels = []
    for order in orderings:
        trans_payoffs_r = transform(norm_payoffs_r, order)
        trans_payoffs_c = transform(norm_payoffs_c, order)
        trans_freq = df_train_truth.apply(
            lambda x: x[list(order)], axis=1).values
        augmented_data.append(np.array([trans_payoffs_r, trans_payoffs_c]))
        augmented_labels.append(trans_freq)

    game_data = np.array(augmented_data).transpose(
        [0, 2, 1, 3, 4]).reshape(250 * 6, 2, 3, 3)
    game_labels = np.array(augmented_labels).reshape(250 * 6, 3)

    # Get desired channels
    game_data = expand_channels(game_data, channels=args.channel)

    # Shuffle data first
    game_ix = list(range(len(game_data)))
    np.random.shuffle(game_ix)
    game_data = game_data[game_ix]
    game_labels = game_labels[game_ix]

    # Prepare splits
    splits = get_splits(game_data, game_labels, args.k_folds)

    criterion = nn.KLDivLoss(
        reduction='sum') if args.loss == 'kl' else nn.MSELoss()

    split_quadratic_losses = []
    split_accuracies = []

    for ix, split in enumerate(splits):
        train_val_data = split[0]
        train_val_labels = split[1]
        test_data = split[2]
        test_labels = split[3]

        model = ConvNetBig(log_softmax=False, softmax=True,
                           dropout=args.dropout, channels=game_data.shape[1])

        model.to(args.device)
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        dataloader = load_game_data(
            train_val_data, train_val_labels, batch_size=args.batch_size,
            seed=args.seed, device=args.device, shuffle=True)

        dataloader_val = load_game_data(train_val_data, train_val_labels, batch_size=1,
                                        seed=args.seed, device=args.device, shuffle=False)
        losses, acc = train(model, criterion, dataloader, dataloader_val, epochs=args.epochs,
                            log_targets=False, model_softmax=True, log=args.log,
                            val_data=test_data, val_labels=test_labels, seed=args.seed, device=args.device)

        dataloader_val = load_game_data(test_data, test_labels, batch_size=1,
                                        seed=args.seed, device=args.device, shuffle=False)
        batch_ql, acc, predictions = evaluate(model, dataloader_val)

        split_quadratic_losses.append(batch_ql)
        split_accuracies.append(acc)

        print('Test split quadratic loss: {}, accuracy: {}'.format(batch_ql, acc))

        torch.save(model.state_dict(), '{}/model_{}_{}_s={}_e={}_split={}.pt'.format(
            args.save_model_path, args.channel, args.loss, args.seed, args.epochs, ix
        ))

    print('Average quadratic loss: {}'.format(np.mean(split_quadratic_losses)))
    print('Average accuracy: {}'.format(np.mean(split_accuracies)))
