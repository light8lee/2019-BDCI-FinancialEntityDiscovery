name=roberta_ext_v25
CUDA_VISIBLE_DEVICES=3 python predict.py \
    --cuda \
    --vocab=bert_wwm_ext/vocab.txt \
    --model=best,step5000,step6000 \
    --data=round2_inputs/ner_T3_L510_F8_LM \
    --save_dir=outputs/${name} \
    --config=outputs/${name}/model_config.json \
    --batch_size=16 \
    2> outputs/${name}/predict.log > outputs/${name}/predict.info &
