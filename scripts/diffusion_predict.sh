output_name=v28-v20-v16-v13-v11-wv1
output_dir=outputs/${output_name}
mkdir -p ${output_dir}
cp scripts/diffusion_predict.sh ${output_dir}
CUDA_VISIBLE_DEVICES=3 python diffusion_predict.py \
    --cuda \
    --min_count=2 \
    --names=roberta_ext_v28,roberta_ext_v20,roberta_ext_v16,roberta_ext_v13,roberta_ext_v11,roberta_wlarge_v1 \
    --data=round2_inputs/ner_T3_L510_F8_LM \
    --save_name=${output_dir}/submit.csv \
    --batch_size=4 \
    2> ${output_dir}/predict.log > ${output_dir}/predict.info &
