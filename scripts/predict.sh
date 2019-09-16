name=bert_wwm_v8
CUDA_VISIBLE_DEVICES=1 python predict.py \
    --cuda \
    --model=epoch4 \
    --data=inputs \
    --save_dir=outputs/${name} \
    --config=outputs/${name}/model_config.json \
    --batch_size=256 \
    2> outputs/${name}/predict.log > outputs/${name}/predict.info &
