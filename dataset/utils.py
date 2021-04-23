import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class LoadDict():
    def __init__(self, Dict):
        self.__dict__.update(Dict)


class KaggleHouse():
    def __init__(self, dcfg):
        if isinstance(dcfg, dict):
            dcfg = LoadDict(dcfg)
        self.dcfg = dcfg

    def read(self, path="../data/"):
        dcfg = self.dcfg
        train_data = pd.read_csv(path + dcfg.train)
        test_data = pd.read_csv(path + dcfg.test)
        print(f"train_data.shape: {train_data.shape}")
        print(f"test_data.shape: {test_data.shape}")
        
        all_features = pd.concat((train_data.iloc[:, 1:],
                                  test_data.iloc[:, 1:]), sort=False)

        # 删除不必要的信息
        print(f"remove {dcfg.rmfeas} from data")
        all_features = all_features.drop(columns=dcfg.rmfeas)
        
        # 归一化数据
        numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
        all_features[numeric_features] = all_features[numeric_features].apply(
                                             lambda x: (x - x.mean()) / (x.std()))
        # 将缺失值设置为平局值（0）
        all_features[numeric_features] = all_features[numeric_features].fillna(0)
        
        print(f"all_features.shape before: {all_features.shape}")
        all_features = pd.get_dummies(all_features, dummy_na=True)
        print(f"all_features.shape after: {all_features.shape}")
        
        n_train = train_data.shape[0]
        train_features = all_features[:n_train]
        test_features = all_features[n_train:]
        train_labels = train_data[dcfg.res_title]
        return train_features, train_labels, test_features

    @staticmethod
    def load_features(fpath, lpath=None):
        """
        ！！！Load出来的精度不够，只有小数点4位！！！
        @fpath: pickle features path
        @lpath: pickle labels path
        """
        print(f"Loading {fpath} ...")
        features = pd.read_pickle(fpath).values
        features = torch.tensor(features, dtype=torch.float32)
        if lpath is not None:
            print(f"Loading {lpath} ...")
            labels = pd.read_pickle(lpath).values
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = None
        return features, labels
    
    @staticmethod
    def save_features(train_features, train_labels, test_features):
        train_features.to_pickle ('train_features.pkl')
        train_labels.to_pickle('train_labels.pkl')
        test_features.to_pickle('test_features.pkl')
    
    def save_submission(self, preds):
        test_data = {}
        test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
        submission = pd.concat([self.test_data['Id'], test_data['Sold Price']], axis=1)
        submission.to_csv('submission.csv', index=False)


class KaggleHouseDataset(Dataset):
    def __init__(self, features, labels=None):
        if labels is not None:
            assert len(features) == len(labels)
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        return (feature, label)