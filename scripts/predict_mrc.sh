name=
CUDA_VISIBLE_DEVICES=2 python predict_mrc.py \
    --cuda \
    --vocab=roberta_zh_ext/vocab.txt \
    --model=best \
    --data=round2_inputs/ner_mrc_T3_L510_F8_NoLM \
    --save_dir=outputs/${name} \
    --config=outputs/${name}/mrc_model_config.json \
    --batch_size=16 \
    2> outputs/${name}/predict.log > outputs/${name}/predict.info &
