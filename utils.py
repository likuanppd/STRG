from copy import deepcopy
import torch
from deeprobust.graph.utils import *


def train(model, epochs, optim, adj, run, features, labels, idx_train, idx_val, idx_test, loss, verbose=True):
    best_loss_val = 9999
    best_acc_val = 0
    weights = deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        logits = model(adj, features)
        l = loss(logits[idx_train], labels[idx_train])
        optim.zero_grad()
        l.backward()
        optim.step()
        acc = evaluate(model, adj, features, labels, idx_val)
        val_loss = loss(logits[idx_val], labels[idx_val])
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
        if acc > best_acc_val:
            best_acc_val = acc
            weights = deepcopy(model.state_dict())
        if verbose:
            if epoch % 10 == 0:
                print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}"
                      .format(epoch, l.item(), acc))
    model.load_state_dict(weights)
    acc = evaluate(model, adj, features, labels, idx_test)
    print("Run {:02d} Test Accuracy {:.4f}".format(run, acc))
    return acc


def train_MLP(model, epochs, optimizer, train_loader, val_loader, test_loader, loss, device, verbose=True):
    model.train()
    best_acc = 0
    best_loss =9999
    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            optimizer.zero_grad()
            l = loss(output, y)
            l.backward()
            optimizer.step()
        n_acc = 0
        loss_total = 0
        n = 0
        best_acc_val = 0
        best_loss_val = 0
        model.eval()
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            n += len(y)
            acc = (pred == y).sum().item()
            n_acc += acc
            l = loss(output, y)
            loss_total += l
        acc_total = n_acc / n
        val_loss = loss_total /n
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
        if acc_total > best_acc_val:
            best_acc_val = acc_total
            weights = deepcopy(model.state_dict())
        if verbose:
            if epoch % 10 == 0:
                print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}"
                      .format(epoch, l.item(), acc_total))
    model.load_state_dict(weights)
    model.eval()
    n_acc = 0
    n = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        pred = torch.argmax(output, dim=1)
        n += len(y)
        acc = (pred == y).sum().item()
        n_acc += acc
    return  n_acc / n


def adj_norm(adj, neighbor_only=False):
    if not neighbor_only:
        adj = torch.add(torch.eye(adj.shape[0]).cuda(), adj)
    if adj.is_sparse:
        degree = adj.to_dense().sum(dim=1)
    else:
        degree = adj.sum(dim=1)
    in_degree_norm = torch.pow(degree.view(1, -1), -0.5).expand(adj.shape[0], adj.shape[0])
    in_degree_norm = torch.where(torch.isinf(in_degree_norm), torch.full_like(in_degree_norm, 0), in_degree_norm)
    out_degree_norm = torch.pow(degree.view(-1, 1), -0.5).expand(adj.shape[0], adj.shape[0])
    out_degree_norm = torch.where(torch.isinf(out_degree_norm), torch.full_like(out_degree_norm, 0), out_degree_norm)
    adj = sparse_dense_mul(adj, in_degree_norm)
    adj = sparse_dense_mul(adj, out_degree_norm)
    return adj


def get_psu_labels(logits, pseudo_labels, idx_train, idx_test, k=30, append_idx=True):
    # idx_train = np.array([], dtype='int32')
    if append_idx:
        idx_train = idx_train
    else:
        idx_train = np.array([], dtype='int64')
    pred_labels = torch.argmax(logits, dim=1)
    pred_labels_test = pred_labels[idx_test]
    for label in range(pseudo_labels.max().item() + 1):
        idx_label = idx_test[pred_labels_test==label]
        logits_label = logits[idx_label][:, label]
        if len(logits_label) > k:
            _, idx_topk = torch.topk(logits_label, k)
        else:
            idx_topk = np.arange(len(logits_label))
        idx_topk = idx_label[idx_topk]
        pseudo_labels[idx_topk] = label
        idx_train = np.concatenate((idx_train, idx_topk))
    return idx_train, pseudo_labels


def evaluate(model, adj, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(adj, features)
        logits = logits[mask]
        test_labels = labels[mask]
        _, indices = logits.max(dim=1)
        correct = torch.sum(indices==test_labels)
        return correct.item() * 1.0 / test_labels.shape[0]