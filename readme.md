# 实战 Kaggle 比赛：预测房价
详见[动手深度学习v2](https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/kaggle-house-price.html)，使用了新数据（[kaggle_house_new](https://www.kaggle.com/c/california-house-prices/overview)）。


## 数据处理
### 无效数据
对于"kaggle_house_new"数据集来说，由于"Address"和"Summary"的内容比较杂乱，因此删除了这两个特征（`rmfeas=['Address', 'Summary']`），还有一些特征也可以删除或者进行处理。

### 标签值（Groundtruth）
由于"Sold Price"（"SalePrice"）值很大，对MSE Loss不友好，因此将标签值取对数（`torch.log1p()`）。这种方式比使用LogRMSE Loss效果好，注意预测时候需要通过`torch.expm1()`还原。

## 其他
### Pickle或Numpy数据文件
本来想将处理后的Pandas数据保存为Pickle（`.pkl`）文件，但是发现浮点数精度不够，而且加载Pickle文件速度很慢（不如直接读取csv文件）

Numpy的`.npz`文件太大（几十个G），没有测试浮点数精度如何。

- [ ] sads 