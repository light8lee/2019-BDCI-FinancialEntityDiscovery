name=roberta_v1
CUDA_VISIBLE_DEVICES=1 python predict.py \
    --cuda \
    --model=best \
    --data=inputs \
    --save_dir=outputs/${name} \
    --config=outputs/${name}/model_config.json \
    --batch_size=32 \
    2> outputs/${name}/predict.log > outputs/${name}/predict.info &
