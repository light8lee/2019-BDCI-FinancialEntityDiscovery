本代码是复赛A榜第七，B榜第五

### 环境

- Python3.6
- Pytorch 1.1.0
- Titan V100
- pytorch_transformers

### 方案介绍

本代码采用了基于序列标注的实体识别和基于机器阅读理解的实体识别相结合的方式。

- 基于序列标注的实体识别，基本结构是`RoBERTa-wwm-ext(finetuned) + BiLSTM + CRF`，同时拼接字特征
[!image1](./resources/tagger.png)
- 基于机器阅读理解的实体识别，参考的是香侬科技的《A Unified MRC Framework for Named Entity Recognition》论文，并自己实现的，同时根据实际情况做了相应的改进。
[!image1](./resources/mrc.png)

具体细节可以参考我们的论文[迎难而上-金融实体-说明论文](./resources/paper.pdf)


### 基于序列标注的实体识别

1. 在初赛和复赛的数据上微调roberta模型（来自讯飞发布的RoBERTa-wwm-ext Chinese，[下载地址](https://drive.google.com/open?id=1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25)，将微调的模型重命名为roberta_finetune_ab2并移动到当前目录下

2.为了和初赛代码兼容，将复赛的训练集和测试集重命名为Train_Data.csv和Test_Data.csv，并保证数据格式为UTF-8,LF结尾，放在round2_data目录下。

3. 数据预处理：

```
bash scripts/create_data.sh
```
会在round2_inputs目录下生成训练、验证和测试集数据

4. 训练NER模型：

```
bash scripts/train.sh
```
会在outputs目录下保存模型

5. NER模型融合:

```
python merge_models.py roberta_ext_v25 best,step5000
```

6. NER模型预测:

```
bash scripts/predict.sh
```

7. NER模型后处理（需要保证submits目录存在）：

```
python postprocess.py --crf_model=roberta_ext_v25
```
会在submits目录下生成roberta_ext_v25.csv
复赛A榜最终基础的模型是这个，分数是0.507



### 基于机器阅读理解的实体识别

1. MRC数据预处理

```
bash scripts/create_mrc_data.sh
```

2. 训练MRC:

```
bash scripts/train_mrc.sh
```

3. MRC模型预测:

```
bash scripts/predict_mrc.sh
```

4. 后处理:

```
python postprocess.py --crf_model=roberta_ext_m11
```
会在submits目录下生成roberta_ext_m11.csv

5.  投票：

```
python vote_submits.py
```

生成11-21.csv文件作为最终结果

