import numpy as np
import matplotlib.pyplot as plt


def get_cam_heatmaps(model, layer, dataloader, indices):
    model.eval()

    samples = []
    labels = []
    heatmaps = []
    predictions = []

    for batch_ix, (data, label) in enumerate(dataloader):
        samples.append(data)
        labels.append(
            np.round(label.detach().cpu().numpy().squeeze(), decimals=3))

    selected_samples = [samples[ix]
                        for ix in range(len(samples)) if ix in indices]
    for sample in selected_samples:
        pred = model(sample)
        target = pred.argmax(dim=1)

        # Get gradient of output wrt model params
        pred[:, target].backward()
        # Pull gradients out of model
        gradients = model.get_activations_gradient()

        # Pool grads across channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = model.get_activations(data)
        # Get activations of last conv layer
        activations = activations[layer].detach()
        # Weight channel activations by gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        # Calculate heatmap valules
        heatmap = torch.mean(activations, dim=1).squeeze()

        # ReLU on heatmap
        zeros = torch.zeros(heatmap.shape).cuda()
        heatmap = torch.max(heatmap, zeros)  # ReLU on heatmap
        heatmap /= torch.max(heatmap)  # Normalize
        heatmaps.append(heatmap)
        predictions.append(pred.detach().cpu().numpy().squeeze())
    return heatmaps, [labels[ix] for ix in range(len(labels)) if ix in indices], predictions


def visualize_heatmaps(heatmaps, labels, predictions, data, show_payoffs=False, diff_data=None, show_diffs=False):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0)

    selected_data = data

    for ix, ax in enumerate(axes.flat):
        viz = ax.matshow(heatmaps[ix].squeeze().detach().cpu().numpy())
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # ax.text('test', ha='center', va='center')
        if show_payoffs:
            for i in range(selected_data.shape[2]):
                for j in range(selected_data.shape[2]):
                    color = 'w' if heatmaps[ix][i,
                                                j] < 0.4 else 'b'  # visibility
                    text = ax.text(j, i, '({}, {})'.format(int(selected_data[ix][0][i, j] * 100),
                                                           int(selected_data[ix][1][i, j] * 100)),
                                   ha="center", va="center", color=color)

        elif show_diffs:
            for i in range(diff_data.shape[2]):
                for j in range(diff_data.shape[2]):
                    color = 'w' if heatmaps[ix][i,
                                                j] < 0.4 else 'b'  # visibility
                    text = ax.text(j, i, '({:.2f}, {:.2f})'.format(diff_data[ix][0][i, j],
                                                                   diff_data[ix][1][i, j]),
                                   ha="center", va="center", color=color, fontsize=7)

        pred_str = [np.round(pred, 3) for pred in predictions[ix]]
        ax.title.set_text('{}\n{}'.format(labels[ix], pred_str))

    fig.colorbar(viz, ax=axes.ravel().tolist())
    plt.show()
