# ema decay ablation
######
blr="5e-4"
ema_decay_init="0.0"
feature_layer="11"
ema_decay_final='0.0'

CUDA_VISIBLE_DEVICES=0 python main_bootstrapped_pretrain.py \
--blr "$blr" \
--ema_decay_init "$ema_decay_init" \
--feature_layer "$feature_layer" \
--ln_feature \
--ema_decay_final "$ema_decay_final" \
--output_dir ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"

CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_finetune/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"

CUDA_VISIBLE_DEVICES=0 python main_linprobe.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_linprobe/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"
######

# ema decay ablation
######
blr="5e-4"
ema_decay_init="0.0"
feature_layer="11"
ema_decay_final='0.5'

CUDA_VISIBLE_DEVICES=0 python main_bootstrapped_pretrain.py \
--blr "$blr" \
--ema_decay_init "$ema_decay_init" \
--feature_layer "$feature_layer" \
--ln_feature \
--ema_decay_final "$ema_decay_final" \
--output_dir ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"

CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_finetune/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"

CUDA_VISIBLE_DEVICES=0 python main_linprobe.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_linprobe/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"
######


