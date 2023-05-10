# """
# MLP self-training
# """
from deeprobust.graph.utils import *
import argparse
from utils import *
import scipy
from model.MLP import MLP
from model.GCN import GCN
import torch.utils.data as Data
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('--ptb_rate', type=float, default=0.2,  help='pertubation rate')
parser.add_argument('--attack', type=str, default='mettack',  help='attack model')
parser.add_argument('--dataset', type=str, default='citeseer',  help='dataset')
parser.add_argument('--seed', type=int, default=20, help='random seed')
parser.add_argument('--k', type=int, default=80, help='the number of pseudo labels')

args = parser.parse_args()
# random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

# loading data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = './ptb_graphs/%s' % (args.attack)
features = scipy.sparse.load_npz('%s/%s_features.npz' % (path, args.dataset))
features = sparse_mx_to_torch_sparse_tensor(features).to_dense()
labels = np.load('%s/%s_labels.npy' % (path, args.dataset))
labels = torch.LongTensor(labels)
perturbed_adj = torch.load('%s/%s_%s%s.pt' % (path, args.attack, args.dataset, str(args.ptb_rate)))
idx_train = np.load('%s/%s_%sidx_train.npy' % (path, args.attack, args.dataset + str(args.ptb_rate)))
idx_val = np.load('%s/%s_%sidx_val.npy' % (path, args.attack, args.dataset + str(args.ptb_rate)))
idx_test = np.load('%s/%s_%sidx_test.npy' % (path, args.attack, args.dataset + str(args.ptb_rate)))

# Hyper-parameters
epochs = 200
n_hidden = 1024
dropout = 0.5
weight_decay = 5e-4
lr = 1e-2
loss = nn.CrossEntropyLoss()
n_class = labels.max().item() + 1
batch_size = 64

# dataloaders
train_dataset = Data.TensorDataset(features[idx_train], labels[idx_train])
train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = Data.TensorDataset(features[idx_val], labels[idx_val])
val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = Data.TensorDataset(features[idx_test], labels[idx_test])
test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model = MLP(features.shape[1], n_class, n_hidden)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == '__main__':
    acc = train_MLP(model, epochs, optimizer, train_loader, val_loader, test_loader, loss, device)
    print('Accuracy:%f' % acc)
    logits = model(features.to(device)).cpu()
    pseudo_labels = labels.clone()
    print(len(idx_train))
    idx_train, pseudo_labels = get_psu_labels(logits, pseudo_labels, idx_train, idx_test, k=args.k, append_idx=False)
    print(len(idx_train))
    acc_total = []
    perturbed_adj = normalize_adj_tensor(perturbed_adj, sparse=True)
    for run in range(10):
        model = GCN(features.shape[1], 16, n_class)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        perturbed_adj = perturbed_adj.to(device)
        features = features.to(device)
        pseudo_labels = pseudo_labels.to(device)
        labels = labels.to(device)
        acc = train(model, epochs, optimizer, perturbed_adj, run, features, pseudo_labels, idx_train, idx_val, idx_test,
                    loss, verbose=False)
        acc_total.append(acc)
    print('Mean Accuracy:%f' % np.mean(acc_total))
    print('Standard Deviation:%f' % np.std(acc_total, ddof=1))
