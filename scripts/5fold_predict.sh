name=bert_wwm_vdh
CUDA_VISIBLE_DEVICES=1 python kfold_predict.py \
    --cuda \
    --vocab=bert_wwm_ext/vocab.txt \
    --data=inputs/ner_full500 \
    --save_dir=outputs/${name} \
    --config=outputs/${name}/model_config.json \
    --batch_size=32 \
    2> outputs/${name}/predict.log > outputs/${name}/predict.info &
    # --models= \
