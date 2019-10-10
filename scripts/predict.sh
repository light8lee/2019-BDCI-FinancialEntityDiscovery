name=bert_wwm_v41
CUDA_VISIBLE_DEVICES=3 python predict.py \
    --cuda \
    --vocab=bert_wwm_ext/vocab.txt \
    --model=best \
    --data=inputs/ner_full510v3 \
    --save_dir=outputs/${name} \
    --config=outputs/${name}/model_config.json \
    --batch_size=32 \
    2> outputs/${name}/predict.log > outputs/${name}/predict.info &
