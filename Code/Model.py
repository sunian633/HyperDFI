import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import compute_contrast_loss, cross_entropy_loss, row_l2_normalize

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.dEmbeds = nn.Parameter(init(t.empty(args.drug, args.latdim)))
        self.fEmbeds = nn.Parameter(init(t.empty(args.food, args.latdim)))

        self.gcnLayer = GCNLayer()
        self.hgnnLayer = HGNNLayer()
        self.classifierLayer = ClassifierLayer()

        if args.dense:
            self.dHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
            self.fHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))

        self.edgeDropper = SpAdjDropEdge()

    def compute_losses(self, drugs, foods, labels, adj, keepRate):
        embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(adj, keepRate)
        dEmbeds, fEmbeds = embeds[:args.drug], embeds[args.drug:]

        dEmbeds = dEmbeds[drugs]
        fEmbeds = fEmbeds[foods]

        pre = self.classifierLayer(dEmbeds, fEmbeds)
        ceLoss = cross_entropy_loss(pre, labels)

        sslLoss = 0
        for i in range(1, args.gnn_layer + 1, 1):
            embeds1 = gcnEmbedsLst[i].detach()
            embeds2 = hyperEmbedsLst[i]
            sslLoss += compute_contrast_loss(
                embeds1[:args.drug], embeds2[:args.drug], t.unique(drugs), args.temp)
            sslLoss += compute_contrast_loss(
                embeds1[args.drug:], embeds2[args.drug:], t.unique(foods), args.temp)

        return ceLoss, sslLoss

    def predict_scores(self, adj, drugs, foods):
        embeds, _, _ = self.forward(adj, 1.0)
        dEmbeds, fEmbeds = embeds[:args.drug], embeds[args.drug:]

        dEmbeds = dEmbeds[drugs]
        fEmbeds = fEmbeds[foods]

        return self.classifierLayer(dEmbeds, fEmbeds)

    def forward(self, adj, keepRate):
        embeds = t.concat([self.dEmbeds, self.fEmbeds], axis=0)
        embedsLst = [embeds]
        gcnEmbedsLst = [embeds]
        hyperEmbedsLst = [embeds]

        ddHyper = self.dEmbeds * args.mult
        ffHyper = self.fEmbeds * args.mult

        if args.dense:
            ddHyper = self.dEmbeds @ self.dHyper
            ffHyper = self.fEmbeds @ self.fHyper

        for _ in range(args.gnn_layer):
            gcnEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), embedsLst[-1])

            hyperDEmbeds = self.hgnnLayer(ddHyper, embedsLst[-1][:args.drug])
            hyperFEmbeds = self.hgnnLayer(ffHyper, embedsLst[-1][args.drug:])
            hyperEmbeds = t.concat([hyperDEmbeds, hyperFEmbeds], axis=0)

            gcnEmbedsLst.append(gcnEmbeds)
            hyperEmbedsLst.append(hyperEmbeds)
            embedsLst.append(gcnEmbeds + hyperEmbeds)

        embeds = sum(embedsLst)
        return embeds, gcnEmbedsLst, hyperEmbedsLst


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return row_l2_normalize(t.spmm(adj, embeds))


class HGNNLayer(nn.Module):
    def __init__(self):
        super(HGNNLayer, self).__init__()

    def forward(self, adj, embeds):
        lat = adj.T @ embeds
        ret = adj @ lat
        return row_l2_normalize(ret)


class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        newVals = vals[mask] / keepRate
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class ClassifierLayer(nn.Module):
    def __init__(self):
        super(ClassifierLayer, self).__init__()
        self.lin1 = nn.Linear(args.latdim * 2, 128)
        self.lin2 = nn.Linear(128, args.num_classes)

    def forward(self, dEmbeds, fEmbeds):
        embeds = t.concat((dEmbeds, fEmbeds), 1)
        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        return self.lin2(embeds)
