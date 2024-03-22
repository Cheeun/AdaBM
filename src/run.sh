DATA_DIR=/mnt/disk1/cheeun914/datasets/
scale=4

# sh run.sh edsr 5 4 4
edsr() {
    CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model EDSR --scale $scale \
    --n_feats 64 --n_resblocks 16 --res_scale 1.0 \
    --pre_train ../pretrained_model/edsr_baseline_x$scale.pt \
    --epochs 10 --test_every 50 --print_every 10 \
    --batch_size_update 2 --batch_size_calib 16 --num_data 100 --patch_size 384 \
    --data_test Set5 --dir_data $DATA_DIR \
    --quantize_a $2 --quantize_w $3 \
    --quantizer 'minmax' --ema_beta 0.9 --quantizer_w 'omse' \
    --lr_w 0.01 --lr_a 0.01 --lr_measure_img 0.1 --lr_measure_layer 0.01 \
    --w_bitloss 50.0 --imgwise --layerwise --bac \
    --img_percentile 10.0 --layer_percentile 30.0 \
    --save edsrbaseline_x$scale/w$3a$2-ours_temp4 \
    --fq \
    --seed 1 \
    # 
}

# sh run.sh edsr_eval 4 4 4
edsr_eval() {
    CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model EDSR --scale $scale \
    --n_feats 64 --n_resblocks 16 --res_scale 1.0 \
    --data_test Set5 --dir_data $DATA_DIR \
    --quantize_a $2 --quantize_w $3 \
    --imgwise --layerwise \
    --test_only \
    --save edsrbaseline_x$scale/w$3a$2-ours_temp3_sd1_test \
    --pre_train ../experiment/edsrbaseline_x$scale/w$3a$2-ours_temp3_sd1/model/checkpoint.pt \
    --fq \
    --test_patch --test_patch_size 96 --test_step_size 96 \
    # --save_results \
    # 
}

# sh run.sh edsr_eval_own 4 4 4
edsr_eval_own() {
    CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model EDSR --scale $scale \
    --n_feats 64 --n_resblocks 16 --res_scale 1.0 \
    --data_test Set5 --dir_data $DATA_DIR \
    --quantize_a $2 --quantize_w $3 \
    --imgwise --layerwise \
    --test_only \
    --save edsrbaseline_x$scale/w$3a$2-ours_temp3_sd1_test \
    --pre_train ../experiment/edsrbaseline_x$scale/w$3a$2-ours_temp3_sd1/model/checkpoint.pt \
    --fq \
    --test_patch --test_patch_size 96 --test_step_size 96 \
    --test_own './../../PTQ-SR/src/figure/0014_bicubic.jpg' \
    # --save_results \
    # --test_own './../../PTQ-SR/src/figure/1459_original.png' \
}

# sh run.sh rdn 4 4 4
rdn(){
    CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model RDN --scale $scale \
    --pre_train ../pretrained_model/rdn_baseline_x$scale.pt \
    --epochs 10 --test_every 50 --print_every 10 \
    --batch_size_update 2 --batch_size_calib 16 --num_data 100 --patch_size 256 \
    --data_test Set5 --dir_data $DATA_DIR \
    --quantize_a $2 --quantize_w $3 \
    --quantizer 'minmax' --ema_beta 0.9 --quantizer_w 'omse' \
    --lr_w 0.01 --lr_a 0.01 --lr_measure_img 0.1 --lr_measure_layer 0.01 \
    --w_bitloss 50.0 --imgwise --layerwise --bac \
    --img_percentile 10.0 --layer_percentile 30.0 \
    --save rdn_x$scale/w$3a$2-ours_temp \
    --fq \
    --seed 1 \
    # 
}

rdn_eval(){
    CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model RDN --scale $scale \
    --data_test Set5 --dir_data $DATA_DIR \
    --quantize_a $2 --quantize_w $3 \
    --imgwise --layerwise \
    --test_only \
    --save rdn_x$scale/w$3a$2-ours_temp_sd1_test \
    --pre_train ../experiment/rdn_x$scale/w$3a$2-ours_temp_sd1/model/model_latest.pt \
    --fq \
    --test_patch --test_patch_size 96 --test_step_size 96 \
    --save_results \
    # 
}

# sh run.sh srresnet 4 4 4
srresnet() {
    CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model SRResNet --scale $scale \
    --n_feats 64 --n_resblocks 16 --res_scale 1.0 \
    --pre_train ../pretrained_model/bnsrresnet_x$scale.pt \
    --epochs 10 --test_every 50 --print_every 10 \
    --batch_size_update 2 --batch_size_calib 16 --num_data 100 --patch_size 384 \
    --data_test Set5 --dir_data $DATA_DIR \
    --quantize_a $2 --quantize_w $3 \
    --quantizer 'minmax' --ema_beta 0.9 --quantizer_w 'omse' \
    --lr_w 0.01 --lr_a 0.01 --lr_measure_img 0.1 --lr_measure_layer 0.01 \
    --w_bitloss 50.0 --imgwise --layerwise --bac \
    --img_percentile 10.0 --layer_percentile 30.0 \
    --save srresnet_x$scale/w$3a$2-ours_temp \
    --fq \
    --seed 1 \
    # 
}

# sh run.sh srresnet_eval 4 4 4
srresnet_eval() {
    CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model SRResNet --scale $scale \
    --n_feats 64 --n_resblocks 16 --res_scale 1.0 \
    --data_test Set5 --dir_data $DATA_DIR \
    --quantize_a $2 --quantize_w $3 \
    --imgwise --layerwise \
    --test_only \
    --save srresnet_x$scale/w$3a$2-ours_temp_sd1_test \
    --pre_train ../experiment/srresnet_x$scale/w$3a$2-ours_temp_sd1/model/model_latest.pt \
    --fq \
    --test_patch --test_patch_size 96 --test_step_size 96 \
    --save_results \
    # 
}

"$@"
