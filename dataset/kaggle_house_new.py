# dataset settings
dataset_type = "kaggle_house_new"
data = dict(
    type="kaggle_house_new",
    train=dataset_type+"/train.csv",
    test=dataset_type+"/test.csv",
    rmfeas=['Address', 'Summary'],  # features to be removed
    res_title="Sold Price",
    # [optional] 提前处理好的数据
    path="data/kaggle_house_new",
    train_features="train_features.pkl",
    train_labels="train_labels.pkl",
    test_features="test_features.pkl",
)


def test_save_load():
    kaggle = KaggleHouse(data)
    tf, tl, ts = kaggle.read()
    print(tf.iloc[0].mean())
#     kaggle.save_features(tf, tl, ts)
    
    fpath = '../data/kaggle_house_new/train_features.pkl'
    tf, tl = KaggleHouse.load_features(fpath)
    import torch
    print(torch.mean(tf[0]))

    
if __name__ == '__main__':
    import os, sys
    os.chdir(sys.path[0])
    
    from utils import KaggleHouse
    
    print(f"dataset_type: {dataset_type}")
#     kaggle.save_features(*kaggle.read())
    
    test_save_load()
    

    
