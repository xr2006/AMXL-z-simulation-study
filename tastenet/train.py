import copy

import torch
import torch.nn.functional as F


def train(model, data_train, data_dev, lr, weight_decay, num_epochs, patience=10):
    """Adam with an L2 penalty (weight decay) and early stopping on the development NLL.
    Returns (best_model, success). success is False when the run is abandoned because a bad
    initialization left it far from any solution."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_model, best_dev, no_improve = None, float('inf'), 0
    for epoch in range(1, num_epochs + 1):
        train_epoch(model, data_train, optimizer)
        loss_train = evaluate_epoch(model, data_train)
        loss_dev = evaluate_epoch(model, data_dev)
        if epoch > 20 and loss_train > 8:
            return None, False
        if loss_dev < best_dev:
            best_dev, best_model, no_improve = loss_dev, copy.deepcopy(model), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return best_model, True


def train_epoch(model, loader, optimizer):
    model.train()
    for batch in loader:
        loss = F.cross_entropy(model(batch['z'], batch['x']), batch['y'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_epoch(model, loader):
    """Mean NLL per observation over the whole loader."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            total += F.cross_entropy(model(batch['z'], batch['x']), batch['y'], reduction='sum').item()
            n += len(batch['y'])
    return total / n
