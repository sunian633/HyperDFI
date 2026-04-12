import os
import random

import numpy as np
import torch as t
import torch.nn.functional as F
import wandb
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, roc_auc_score

import Utils.RunLogger as logger
from DataLoader import DataLoader
from Model import Model
from Params import args
from Utils.RunLogger import log
from Utils.Utils import *


def seed_everything(seed):
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)

    if t.cuda.is_available():
        t.cuda.manual_seed(seed)
        t.backends.cudnn.benchmark = False
        t.backends.cudnn.deterministic = True


class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('DRUG', args.drug, 'FOOD', args.food)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Acc', 'Precision', 'AUC', 'AUPR', 'F1']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def run_external_eval(self):
        self.setup_model()
        log('Model Prepared')
        if args.load_model is not None:
            self.restore_model()
        reses = self.run_test_epoch()
        log(self.format_metrics('Test', args.epoch, reses, True))
        return reses['Acc']

    def format_metrics(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        return ret[:-2] + '  '

    def restore_model(self):
        self.model.load_state_dict(t.load('../Models/' + args.load_model + '.pkl'))
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        log('Model Loaded')

    def run(self):
        self.setup_model()
        log('Model Prepared')
        if args.load_model is not None:
            self.restore_model()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.run_train_epoch()
            train_loss = reses
            log(self.format_metrics('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.run_test_epoch()
                test_r = reses
                log(self.format_metrics('Test', ep, reses, tstFlag))
            logs = {'loss_all': train_loss['Loss'], 'loss_pre': train_loss['preLoss'],
                    'test_acc': test_r['Acc'], 'test_precision': test_r['Precision'],
                    'test_auc': test_r['AUC'], 'test_aupr': test_r['AUPR'], 'test_f1': test_r['F1']}

        reses = self.run_test_epoch()
        log(self.format_metrics('Test', args.epoch, reses, True))
        return reses['Acc']

    def persist_model(self, model_path):
        model_parent_path = os.path.join(wandb.run.dir, 'ckl')
        if not os.path.exists(model_parent_path):
            os.mkdir(model_parent_path)
        t.save(self.model.state_dict(), '{}/{}_model.pkl'.format(model_parent_path, model_path))

    def setup_model(self):
        self.model = Model().cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    def run_train_epoch(self):
        self.model.train()
        trnLoader = self.handler.trnLoader
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        for tem in trnLoader:
            drugs, foods, labels = tem
            drugs = drugs.long().cuda()
            foods = foods.long().cuda()
            labels = labels.long().cuda()

            ceLoss, sslLoss = self.model.compute_losses(
                drugs, foods, labels, self.handler.torchBiAdj, args.keepRate)
            sslLoss = sslLoss * args.ssl_reg

            regLoss = compute_reg_loss(self.model) * args.reg
            loss = ceLoss + regLoss + sslLoss
            loss = ceLoss + regLoss
            epLoss += loss.item()
            epPreLoss += ceLoss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret

    def run_test_epoch(self):
        self.model.eval()
        tstLoader = self.handler.tstLoader
        all_preds = []
        all_labels = []
        all_probs = []

        with t.no_grad():
            for tem in tstLoader:
                drugs, foods, labels = tem
                drugs = drugs.long().cuda()
                foods = foods.long().cuda()
                labels = labels.long().cuda()
                pre = self.model.predict_scores(self.handler.torchBiAdj, drugs, foods)
                pre_prob = F.sigmoid(pre).detach().cpu().numpy()
                pos_prob = pre_prob[:, 1]
                pre_label = pre.data.max(1, keepdim=True)[1].detach().cpu().numpy()
                true_label = labels.detach().cpu().numpy()

                all_preds.extend(pre_label.flatten())
                all_labels.extend(true_label.flatten())
                all_probs.extend(pos_prob)

        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        epAcc = accuracy_score(all_labels, all_preds)
        epPrecision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        epF1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

        try:
            epAUC = roc_auc_score(all_labels, all_probs)
        except:
            epAUC = 0.0
        try:
            epAUPR = average_precision_score(all_labels, all_probs)
        except:
            epAUPR = 0.0

        ret = dict()
        ret['Acc'] = epAcc
        ret['Precision'] = epPrecision
        ret['F1'] = epF1
        ret['AUC'] = epAUC
        ret['AUPR'] = epAUPR
        return ret


if __name__ == '__main__':
    use_cuda = args.gpu >= 0 and t.cuda.is_available()
    device = 'cuda:{}'.format(args.gpu) if use_cuda else 'cpu'
    if use_cuda:
        t.cuda.set_device(device)
    args.device = device

    logger.saveDefault = True

    log('Start')
    loader = DataLoader()
    loader.load_data()
    log('Load Data')

    coach = Coach(loader)
    config = dict()
    results = list()
    for i in range(args.iteration):
        print('{}-th iteration'.format(i + 1))
        seed = args.seed + i
        config['seed'] = seed
        config['iteration'] = i + 1
        seed_everything(seed)
        if args.data == 'LINCS':
            result = coach.run_external_eval()
        else:
            result = coach.run()
        results.append(result)

    avg_r = np.mean(np.array(results), axis=0)
    std_r = np.std(results, axis=0)

    results.append(avg_r)
    results.append(std_r)
