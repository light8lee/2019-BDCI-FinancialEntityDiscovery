name=bert_wwm_v21k1
CUDA_VISIBLE_DEVICES=2 python kfold_predict.py \
    --cuda \
    --data=inputs/ner_full510v1 \
    --save_dir=outputs/${name} \
    --config=outputs/${name}/model_config.json \
    --batch_size=32 \
    2> outputs/${name}/predict.log > outputs/${name}/predict.info &
    # --models= \
