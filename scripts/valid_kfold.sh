name=cnn_diffpool_v1
CUDA_VISIBLE_DEVICES=1 python kfold_validation.py \
    80,78,77,79,76,78,80,80,80,80 \
    --cuda \
    --data=inputs/ \
    --save_dir=outputs/${name} \
    --config=outputs/${name}/model_config.json \
    --batch_size=64 \
    2> outputs/${name}/kfold.log
