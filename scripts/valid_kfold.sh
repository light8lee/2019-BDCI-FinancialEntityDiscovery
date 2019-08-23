name=kfold_lst_sdg1t
CUDA_VISIBLE_DEVICES=2 python kfold_validation.py \
    --cuda \
    --data=inputs_all/stsbenchmark_inputs_dg/ \
    --save_dir=outputs/${name} \
    --config=outputs/${name}/model_config.json \
    --batch_size=64 \
    2> outputs/${name}/kfold.log &
