# level2_tier2_models.py
#
# The tier-2 models as functions, for 09_Transplant_Tier2.py: the dense network (as in 06) and
# the weighted KNN votes (as in 07). 06 and 07 keep their own copies of this code, unchanged.

import copy
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from level2_classifier_common import (CLASSES, RANDOM_STATE, NN_DROPOUT, NN_LEARNING_RATE, NN_BATCH_SIZE,
                                      NN_MAX_EPOCHS, NN_PATIENCE, KNN_K, KNN_QUERY_BATCH)

to_tensor = lambda a: torch.as_tensor(a, dtype=torch.float32)
to_index = lambda labels: torch.as_tensor(pd.Index(CLASSES).get_indexer(labels), dtype=torch.long)


def train_network(X_fit, y_fit, X_val, y_val, class_weights, hidden, log=print):
    """Train the dense network as 06 does: Linear-ReLU-Dropout per hidden layer, class-weighted
    cross-entropy, Adam, early stopping on the (equally weighted) validation loss, keeping the best
    epoch's weights. Returns the network (in eval mode) and a data frame of the loss per epoch."""
    torch.manual_seed(RANDOM_STATE)
    layers, n_in = [], X_fit.shape[1]
    for n_out in hidden:
        layers += [nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(NN_DROPOUT)]
        n_in = n_out
    layers.append(nn.Linear(n_in, len(CLASSES)))
    net = nn.Sequential(*layers)

    Xt_fit, yt_fit = to_tensor(X_fit), to_index(y_fit)
    Xt_val, yt_val = to_tensor(X_val), to_index(y_val)
    loss_fn = nn.CrossEntropyLoss(weight=to_tensor(class_weights))
    optimizer = torch.optim.Adam(net.parameters(), lr=NN_LEARNING_RATE)
    history, best_val, best_state, best_epoch = [], np.inf, None, 0
    for epoch in range(1, NN_MAX_EPOCHS + 1):
        net.train()
        order = torch.randperm(len(yt_fit))
        total = 0.
        for start in range(0, len(order), NN_BATCH_SIZE):
            batch = order[start:start + NN_BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(net(Xt_fit[batch]), yt_fit[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(Xt_val), yt_val).item()
        history.append(dict(epoch=epoch, train_loss=total / len(yt_fit), val_loss=val_loss))
        # Early stopping: keep the weights with the lowest validation loss
        if val_loss < best_val:
            best_val, best_state, best_epoch = val_loss, copy.deepcopy(net.state_dict()), epoch
        elif epoch - best_epoch >= NN_PATIENCE:
            break
        if epoch % 10 == 0:
            log(f"epoch {epoch}: training loss {history[-1]['train_loss']:.4f}, validation loss {val_loss:.4f}")
    net.load_state_dict(best_state)
    net.eval()
    log(f"stopped after {epoch} epochs; best validation loss {best_val:.4f} at epoch {best_epoch}")
    return net, pd.DataFrame(history)


def network_proba(net, X):
    """Class probabilities from the network (columns in the order of CLASSES)."""
    with torch.no_grad():
        return torch.softmax(net(to_tensor(X)), dim=1).numpy().astype(np.float64)


def knn_class_vote_sums(Xq, X_ref, ref_class):
    """As in 07: for every k in KNN_K, an (n_query x n_classes) array of each class's 1/distance-weighted
    votes among the k nearest reference cells. Xq, X_ref: float32 tensors; ref_class: long tensor."""
    k_max = max(KNN_K)
    ref_sq = (X_ref ** 2).sum(dim=1)
    sums = {k: np.empty((len(Xq), len(CLASSES)), dtype=np.float64) for k in KNN_K}
    for start in range(0, len(Xq), KNN_QUERY_BATCH):
        q = Xq[start:start + KNN_QUERY_BATCH]
        d2 = (q ** 2).sum(dim=1, keepdim=True) + ref_sq - 2 * q @ X_ref.T
        d2, nn_idx = torch.topk(d2, k_max, dim=1, largest=False, sorted=True)
        # Exact matches (distance 0) get a very large weight, so they dominate, as in scikit-learn
        w = 1 / torch.sqrt(d2.clamp(min=0)).clamp(min=1e-6)
        votes = torch.zeros(len(q), k_max, len(CLASSES))
        votes.scatter_(2, ref_class[nn_idx].unsqueeze(2), w.unsqueeze(2))
        cum = votes.cumsum(dim=1)
        for k in KNN_K:
            sums[k][start:start + len(q)] = cum[:, k - 1, :].double().numpy()
    return sums


def knn_predict(sums_k, ref_freq, gamma):
    """As in 07: class scores (rows sum to 1) and labels, with votes weighted by (1/class frequency)^gamma."""
    scores = sums_k * (1 / ref_freq) ** gamma
    scores = scores / scores.sum(axis=1, keepdims=True)
    return scores, CLASSES[scores.argmax(axis=1)]
