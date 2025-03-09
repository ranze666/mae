# default setting
######
blr="5e-4"
ema_decay_init="0.0"
feature_layer="11"

CUDA_VISIBLE_DEVICES=0 python main_bootstrapped_pretrain.py \
--blr "$blr" \
--ema_decay_init "$ema_decay_init" \
--feature_layer "$feature_layer" \
--ln_feature \
--output_dir ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"

CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_finetune/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"

CUDA_VISIBLE_DEVICES=0 python main_linprobe.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_linprobe/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"
######

# lr
######
blr="2e-4"
ema_decay_init="0.0"
feature_layer="11"

CUDA_VISIBLE_DEVICES=0 python main_bootstrapped_pretrain.py \
--blr "$blr" \
--ema_decay_init "$ema_decay_init" \
--feature_layer "$feature_layer" \
--ln_feature \
--output_dir ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"

CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_finetune/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"

CUDA_VISIBLE_DEVICES=0 python main_linprobe.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_linprobe/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"
######

######
blr="5e-4"
ema_decay_init="0.0"
feature_layer="11"

CUDA_VISIBLE_DEVICES=0 python main_bootstrapped_pretrain.py \
--blr "$blr" \
--ema_decay_init "$ema_decay_init" \
--feature_layer "$feature_layer" \
--ln_feature \
--output_dir ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"

CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_finetune/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"

CUDA_VISIBLE_DEVICES=0 python main_linprobe.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_linprobe/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"
######
