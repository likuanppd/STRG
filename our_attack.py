from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN


seed = 15
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = 'citeseer'
data = Dataset(root='/tmp/', name=dataset, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels

features = sparse_mx_to_torch_sparse_tensor(features)
features = features.to_dense()

adj = sparse_mx_to_torch_sparse_tensor(adj)
adj = adj.to_dense()
adj = adj.to(device)
ptb_rate = 0.2
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
idx_unlabeled = np.union1d(idx_val, idx_test)
n_perturbations = int(adj.sum().item() * ptb_rate / 2)


def compute_lambda(adj, idx_train, idx_test):
    num_all = adj.sum().item() / 2
    train_train = adj[idx_train][:, idx_train].sum().item() / 2
    test_test = adj[idx_test][:, idx_test].sum().item() / 2
    train_test = num_all - train_train - test_test
    return train_train / num_all, train_test / num_all, test_test / num_all


def heuristic_attack(adj, n_perturbations, idx_train, idx_unlabeled, lambda_1, lambda_2):
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_unlabeled)
    degree = adj.sum(dim=1).to('cpu')
    canditate_train_idx = idx_train[degree[idx_train] < (int(degree.mean()) + 1)]
    candidate_test_idx = idx_test[degree[idx_test] < (int(degree.mean()) + 1)]
    #     candidate_test_idx = idx_test
    perturbed_adj = adj.clone()
    cnt = 0
    train_ratio = lambda_1 / (lambda_1 + lambda_2)
    n_train = int(n_perturbations * train_ratio)
    n_test = n_perturbations - n_train
    while cnt < n_train:
        node_1 = np.random.choice(canditate_train_idx, 1)
        node_2 = np.random.choice(canditate_train_idx, 1)
        if labels[node_1] != labels[node_2] and adj[node_1, node_2] == 0:
            perturbed_adj[node_1, node_2] = 1
            perturbed_adj[node_2, node_1] = 1
            cnt += 1

    cnt = 0
    while cnt < n_test:
        node_1 = np.random.choice(canditate_train_idx, 1)
        node_2 = np.random.choice(candidate_test_idx, 1)
        if labels[node_1] != labels[node_2] and perturbed_adj[node_1, node_2] == 0:
            perturbed_adj[node_1, node_2] = 1
            perturbed_adj[node_2, node_1] = 1
            cnt += 1
    return perturbed_adj


# Generate perturbations
lambda_1, lambda_2, lambda_3 = compute_lambda(adj, idx_train, idx_unlabeled)
perturbed_adj = heuristic_attack(adj, n_perturbations, idx_train, idx_unlabeled, lambda_1, lambda_2)

model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
model = model.to(device)
labels = torch.LongTensor(labels)

model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)

model.eval()
# You can use the inner function of model to test
model.test(idx_test)


