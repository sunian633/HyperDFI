import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import pandas as pd
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader


POSITIVE_RELATION = 1.0


class DataLoader:
    def __init__(self):
        self.data = args.data

    def remap_entity_ids(self, raw_ids):
        unique_ids = list(set(raw_ids))
        id_map = {old: new for new, old in enumerate(sorted(unique_ids))}
        mapped_ids = np.array([id_map[x] for x in raw_ids])
        return mapped_ids, id_map, len(unique_ids)

    def normalize_adj_matrix(self, mat):
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

    def load_interaction_data(self, dataset, mode='transductive', testing=True, relation_map=None,
                              post_relation_map=None):
        dtypes = {
            'drug_nodes': np.str_, 'food_nodes': np.int32,
            'relations': np.float32}

        filename_train = '../Data/' + dataset + '/' + mode + '/train.csv'
        filename_test = '../Data/' + dataset + '/' + mode + '/test.csv'

        data_train = pd.read_csv(
            filename_train, header=None,
            names=['drug_nodes', 'food_nodes', 'relations'], dtype=dtypes)

        data_test = pd.read_csv(
            filename_test, header=None,
            names=['drug_nodes', 'food_nodes', 'relations'], dtype=dtypes)

        data_array_train = np.array(data_train.values.tolist())
        data_array_test = np.array(data_test.values.tolist())
        data_array = np.concatenate([data_array_train, data_array_test], axis=0)

        drug_nodes_relations = data_array[:, 0].astype(dtypes['drug_nodes'])
        food_nodes_relations = data_array[:, 1].astype(dtypes['food_nodes'])
        relations = data_array[:, 2].astype(dtypes['relations'])

        if relation_map is not None:
            for i, relation in enumerate(relations):
                relations[i] = relation_map[relation]

        drug_nodes_relations, drug_id_map, num_drugs = self.remap_entity_ids(drug_nodes_relations)
        food_nodes_relations, food_id_map, num_foods = self.remap_entity_ids(food_nodes_relations)

        drug_nodes_relations = drug_nodes_relations.astype(np.int64)
        food_nodes_relations = food_nodes_relations.astype(np.int32)
        relations = relations.astype(np.float64)

        drug_nodes = drug_nodes_relations
        food_nodes = food_nodes_relations

        neutral_relation = -1
        relation_dict = {r: i for i, r in enumerate(np.sort(np.unique(relations)).tolist())}
        if POSITIVE_RELATION not in relation_dict:
            raise ValueError(
                'Positive relation label {} is missing from dataset {}'.format(
                    POSITIVE_RELATION, dataset))

        labels = np.full((num_drugs, num_foods), neutral_relation, dtype=np.int32)
        labels[drug_nodes, food_nodes] = np.array([relation_dict[r] for r in relations])

        for i in range(len(drug_nodes)):
            assert labels[drug_nodes[i], food_nodes[i]] == relation_dict[relations[i]]

        labels = labels.reshape([-1])

        num_train = data_array_train.shape[0]
        num_test = data_array_test.shape[0]
        num_val = int(np.ceil(num_train * 0.2))
        num_train = num_train - num_val

        pairs_nonzero = np.array([[drug, food] for drug, food in zip(drug_nodes, food_nodes)])
        idx_nonzero = np.array([drug * num_foods + food for drug, food in pairs_nonzero])

        for i in range(len(relations)):
            assert labels[idx_nonzero[i]] == relation_dict[relations[i]]

        idx_nonzero_train = idx_nonzero[0:num_train + num_val]
        idx_nonzero_test = idx_nonzero[num_train + num_val:]

        pairs_nonzero_train = pairs_nonzero[0:num_train + num_val]
        pairs_nonzero_test = pairs_nonzero[num_train + num_val:]

        rand_idx = list(range(len(idx_nonzero_train)))
        np.random.seed(42)
        np.random.shuffle(rand_idx)
        idx_nonzero_train = idx_nonzero_train[rand_idx]
        pairs_nonzero_train = pairs_nonzero_train[rand_idx]

        idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
        pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

        val_idx = idx_nonzero[0:num_val]
        train_idx = idx_nonzero[num_val:num_train + num_val]
        test_idx = idx_nonzero[num_train + num_val:]

        assert len(test_idx) == num_test

        val_pairs_idx = pairs_nonzero[0:num_val]
        train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
        test_pairs_idx = pairs_nonzero[num_train + num_val:num_train + num_val + num_test]

        drug_test_idx, food_test_idx = test_pairs_idx.transpose()
        drug_val_idx, food_val_idx = val_pairs_idx.transpose()
        drug_train_idx, food_train_idx = train_pairs_idx.transpose()

        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]

        if True:
            drug_train_idx = np.hstack([drug_train_idx, drug_val_idx])
            food_train_idx = np.hstack([food_train_idx, food_val_idx])
            train_labels = np.hstack([train_labels, val_labels])
            train_idx = np.hstack([train_idx, val_idx])

        class_values = np.sort(np.unique(relations))

        relation_mx_train = np.zeros(num_drugs * num_foods, dtype=np.float32)
        relation_mx_test = np.zeros(num_drugs * num_foods, dtype=np.float32)

        # The classifier is trained with both positive (1) and negative (-1)
        # samples, but the bipartite graph represents observed interactions.
        # Therefore, only positive samples are added as graph edges.
        positive_class = relation_dict[POSITIVE_RELATION]
        positive_train_mask = train_labels == positive_class
        positive_test_mask = test_labels == positive_class
        positive_train_idx = train_idx[positive_train_mask]
        positive_test_idx = test_idx[positive_test_mask]

        relation_mx_train[positive_train_idx] = 1.0
        relation_mx_test[positive_test_idx] = 1.0

        assert np.all(relation_mx_train[train_idx[~positive_train_mask]] == 0.0)
        assert np.all(relation_mx_test[test_idx[~positive_test_mask]] == 0.0)

        relation_mx_train = sp.csr_matrix(relation_mx_train.reshape(num_drugs, num_foods))
        relation_mx_test = sp.csr_matrix(relation_mx_test.reshape(num_drugs, num_foods))

        return relation_mx_train, relation_mx_test, train_labels, drug_train_idx, food_train_idx, \
            val_labels, drug_val_idx, food_val_idx, test_labels, drug_test_idx, food_test_idx, class_values

    def build_torch_biadj(self, mat):
        drug_block = sp.csr_matrix((args.drug, args.drug))
        food_block = sp.csr_matrix((args.food, args.food))
        mat = sp.vstack([sp.hstack([drug_block, mat]), sp.hstack([mat.transpose(), food_block])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalize_adj_matrix(mat)

        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def load_data(self):
        relation_mx_train, relation_mx_test, train_labels, drug_train_idx, food_train_idx, \
            val_labels, drug_val_idx, food_val_idx, test_labels, drug_test_idx, food_test_idx, class_values = \
            self.load_interaction_data(args.data)

        trnMat, tstMat = relation_mx_train, relation_mx_test
        trnMat[trnMat >= 1] = 1
        tstMat[tstMat >= 1] = 1

        if type(trnMat) != coo_matrix:
            trnMat = sp.coo_matrix(trnMat)
        if type(tstMat) != coo_matrix:
            tstMat = sp.coo_matrix(tstMat)
        args.drug, args.food = trnMat.shape
        args.num_classes = len(class_values)
        self.torchBiAdj = self.build_torch_biadj(trnMat)

        trnData = TrnData(train_labels, drug_train_idx, food_train_idx)
        self.trnLoader = dataloader.DataLoader(
            trnData, batch_size=args.batch, shuffle=False, num_workers=0)

        tstData = TstData(test_labels, drug_test_idx, food_test_idx)
        self.tstLoader = dataloader.DataLoader(
            tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)


class TrnData(data.Dataset):
    def __init__(self, train_labels, drug_train_idx, food_train_idx):
        self.train_labels = train_labels
        self.drug_train_idx = drug_train_idx
        self.food_train_idx = food_train_idx

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        return self.drug_train_idx[idx], self.food_train_idx[idx], self.train_labels[idx]


class TstData(data.Dataset):
    def __init__(self, test_labels, drug_test_idx, food_test_idx):
        self.test_labels = test_labels
        self.drug_test_idx = drug_test_idx
        self.food_test_idx = food_test_idx

    def __len__(self):
        return len(self.test_labels)

    def __getitem__(self, idx):
        return self.drug_test_idx[idx], self.food_test_idx[idx], self.test_labels[idx]
