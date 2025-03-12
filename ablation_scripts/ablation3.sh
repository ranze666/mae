

######
blr="5e-4"
ema_decay_init="0.0"
feature_layer="5"

CUDA_VISIBLE_DEVICES=2 python main_bootstrapped_pretrain.py \
--blr "$blr" \
--ema_decay_init "$ema_decay_init" \
--feature_layer "$feature_layer" \
--ln_feature \
--output_dir ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"

CUDA_VISIBLE_DEVICES=2 python main_finetune.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_finetune/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"

CUDA_VISIBLE_DEVICES=2 python main_linprobe.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_linprobe/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"
######



# ln feature
######
blr="5e-4"
ema_decay_init="0.0"
feature_layer="11"

CUDA_VISIBLE_DEVICES=2 python main_bootstrapped_pretrain.py \
--blr "$blr" \
--ema_decay_init "$ema_decay_init" \
--feature_layer "$feature_layer" \
--output_dir ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_noln

CUDA_VISIBLE_DEVICES=2 python main_finetune.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_noln/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_finetune/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_noln

CUDA_VISIBLE_DEVICES=2 python main_linprobe.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_noln/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_linprobe/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_noln
######



# use pixel
######
blr="5e-4"
ema_decay_init="0.0"
feature_layer="11"

CUDA_VISIBLE_DEVICES=2 python main_bootstrapped_pretrain.py \
--blr "$blr" \
--ema_decay_init "$ema_decay_init" \
--feature_layer "$feature_layer" \
--ln_feature \
--use_pixel \
--pixel_loss_decay \
--output_dir ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_usepixel

CUDA_VISIBLE_DEVICES=2 python main_finetune.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_usepixel/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_finetune/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_usepixel

CUDA_VISIBLE_DEVICES=2 python main_linprobe.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_usepixel/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_linprobe/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_usepixel
######


