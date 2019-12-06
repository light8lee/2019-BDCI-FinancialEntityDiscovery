目录结构：
```
Work/
	|transformer-master
	|G4M
	|roberta_zh_ext
	|roberta_finetune_ab2（在微调模型阶段会生成，在训练阶段需要移动到G4M目录下）
```
### 基于序列标注的实体识别
1. 在初赛和复赛的数据上微调roberta模型（来自讯飞发布的RoBERTa-wwm-ext Chinese，[下载地址](https://drive.google.com/open?id=1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25)，放到`Work/roberta_zh_ext`目录下解压，并重命名bert_config.json为config.json，这一步已经完成，可以直接用），进入transformer-master目录下:
    (1)	`bash preprocess.sh`，在data_for_lm目录下会生成a.txt和b.txt
    (2)	手动拷贝b.txt的内容到a.txt，保存为ab.txt
    (3)	微调：`bash run_lm_finetuning.sh`，会在Work目录下生成模型文件夹，需要重命名为roberta_finetune_ab2并移动到G4M目录下（如果不需要从头复现，可以直接使用之前微调好的模型，在G4M目录下，文件夹名称为roberta_finetune_ab2）
2. 进入G4M目录，为了和初赛代码兼容，将复赛的训练集和测试集重命名为Train_Data.csv和Test_Data.csv，并保证数据格式为UTF-8,LF结尾，放在round2_data目录下。
3. 数据预处理：（已放入处理好的数据，可以跳过）
```
bash scripts/create_data.sh
```
会在round2_inputs目录下生成训练、验证和测试集数据
4. 训练NER模型：（如果不需要从头训练，可以直接解压outputs目录下的roberta_ext_v25_bak.tar.gz文件，将解压出来的目录重命名为roberta_ext_v25，并直接到第6步，需要执行的命令：
```
cd outputs
tar zxvf roberta_ext_v25_bak.tar.gz
mv roberta_ext_v25_bak roberta_ext_v25
cd ..
```
）
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
如果不考虑后续步骤，可以使用该模型作为最终结果

### 基于机器阅读理解的实体识别

1. MRC数据预处理
```
bash scripts/create_mrc_data.sh
```
2. 训练MRC:（可以使用已保存的模型，解压outpus目录下的roberta_ext_m11.tar.gz:
```
cd outputs
tar zxvf roberta_ext_m11.tar.gz
cd ..
```
）
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
5.  投票：（为了适应只有两个模型投票，已经更改了vote_submits.py的代码）
```
python vote_submits.py
```
生成11-21.csv文件作为最终结果
