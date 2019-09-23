name=bert_wwm_vd
CUDA_VISIBLE_DEVICES=1 python predict.py \
    --cuda \
    --vocab=bert_wwm_ext/vocab.txt \
    --model=best \
    --data=inputs/ner_full \
    --save_dir=outputs/${name} \
    --config=outputs/${name}/model_config.json \
    --batch_size=64 \
    2> outputs/${name}/predict.log > outputs/${name}/predict.info &
