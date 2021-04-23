import torch
import model
import pandas as pd

from dataset.kaggle_house_new import data as datacfg
from dataset.utils import KaggleHouse

class LoadDict():
    def __init__(self, Dict):
        self.__dict__.update(Dict)


if __name__ == '__main__':
    device = torch.device(f"cuda:1")
    
    net = model.model_v1_8(59520)
    net.load_state_dict(torch.load('net.pt'))
    net.to(device)
    net.eval()
    preds = torch.Tensor()
    
    dcfg = LoadDict(datacfg)
    _, _, test_features = KaggleHouse(dcfg).read('data/')
    test_features = torch.tensor(test_features.values,
                                 dtype=torch.float32)
    test_data = pd.read_csv('data/' + dcfg.test)
    
    preds = net(test_features.to(device))
    preds = torch.expm1(preds).detach().cpu()
    
    # 将其重新格式化以导出到Kaggle
    test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv('submission.csv', index=False)