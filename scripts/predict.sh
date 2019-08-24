name=bert_v1
CUDA_VISIBLE_DEVICES=0 python predict.py \
    QQP \
    --cuda \
    --data=glue_inputs/QQP/ \
    --save_dir=outputs/${name} \
    --config=outputs/${name}/model_config.json \
    --batch_size=64 \
    2> outputs/${name}/predict.log &
