import numpy as np
import pandas as pd
import torch
import time
import model

from torch import nn
from d2l import torch as d2l
from argparse import ArgumentParser
from dataset.utils import KaggleHouse, KaggleHouseDataset


dcfg = "dataset/kaggle_house_new.py"

def get_args():
    parser = ArgumentParser()
#     parser.add_argument('device', help='GPU id')
#     parser.add_argument('config', help='Config file')
#     parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='GPU ID')
    return parser.parse_args()


def init_net(net, device, cfg):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)


def LogRMSELoss(preds, labels):
    e1 = 1e-7
    preds = torch.clamp(preds, 1, float('inf'))
    return torch.sqrt(torch.mean(torch.log(preds/(labels+e1))**2))


def evaluate(net, data_iter, eval_funs, device=None):
    assert isinstance(eval_funs, list)
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    metric = [0.] * len(eval_funs)
    num = 0
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        yhat = net(X)
        num += X.shape[0]
        for i, eval_fun in enumerate(eval_funs):
            metric[i] += eval_fun(yhat, y) * X.shape[0]
    return metric[0] / num


def train(net, train_iter, eval_iter, num_epochs, optimizer, loss, device):
    print('training on', device)
    net.to(device)
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        # 训练损失之和，范例数
        metric = 0.
        num = 0
        net.train()
        ctime = time.time()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                num += X.shape[0]
                metric += l * X.shape[0]
        ctime = time.time() - ctime
        train_loss = torch.sqrt(metric / num)
#             print(f'iter-{i}, train loss: {l:.3f}')
        if eval_iter is not None:
            eval_l = evaluate(net, eval_iter, [loss], device)
            print(f'epoch-{epoch}, train loss: {train_loss:.4f}, \
eval loss: {torch.sqrt(eval_l):.4f}, {ctime:.2f}s/{num}')
        else:
            print(f'epoch-{epoch}, train loss: {train_loss:.4f}, {ctime:.2f}s/{num}')
            
    torch.save(net.state_dict(), "net.pt")


def get_dataset(dcfg):
#     import os
#     trfpath = os.path.join(dcfg.path, dcfg.train_features)
#     trlpath = os.path.join(dcfg.path, dcfg.train_labels)
#     tsfpath = os.path.join(dcfg.path, dcfg.test_features)
    
#     train_features, train_labels = KaggleHouse.load_features(trfpath, trlpath)
#     test_features, _ = KaggleHouse.load_features(tsfpath)
#     print("train number", len(train_features))
#     print("test number", len(test_features))
    
    train_features, train_labels, test_features = KaggleHouse(dcfg).read('data/')
    train_features = torch.tensor(train_features.values,
                              dtype=torch.float32)
    test_features = torch.tensor(test_features.values,
                             dtype=torch.float32)
    train_labels = torch.tensor(train_labels.values.reshape(-1, 1),
                            dtype=torch.float32)
    train_labels = torch.log1p(train_labels)

    train_ds = KaggleHouseDataset(train_features, train_labels)
    test_ds = KaggleHouseDataset(test_features)
    return train_ds, test_ds
    

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
        
class LoadDict():
    def __init__(self, Dict):
        self.__dict__.update(Dict)

if __name__ == '__main__':
    from dataset.kaggle_house_new import data as datacfg
    from torch.utils.data import DataLoader
    import config
    
    args = get_args()
    cfg = LoadDict(config.default)
    dcfg = LoadDict(datacfg)
    print(config.default)
    
    train_dataset, test_dataset = get_dataset(dcfg)
    train_iter = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=1)
    test_iter = DataLoader(test_dataset, cfg.batch_size, shuffle=False, num_workers=1)

    _net = getattr(model, cfg.model)
    net = _net(train_dataset[0][0].shape[0])
    net.apply(weight_init)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.wd)
#     optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#     loss = LogRMSELoss # nn.MSELoss()
    loss = nn.MSELoss()
    
    num_epochs = cfg.num_epochs
    device = torch.device(f"cuda:{args.device}")
     
#     from torch.utils.data import Subset
#     train_num = len(train_dataset)
#     t_ds1 = Subset(train_dataset, range(0, train_num//5*4))
#     e_ds1 = Subset(train_dataset, range(train_num//5*4, train_num))
#     train_iter = DataLoader(t_ds1, cfg.batch_size, shuffle=True, num_workers=1)
#     eval_iter = DataLoader(e_ds1, cfg.batch_size*2, shuffle=False, num_workers=1)
#     train(net, train_iter, eval_iter, num_epochs, optimizer, loss, device)

    train(net, train_iter, None, num_epochs, optimizer, loss, device)