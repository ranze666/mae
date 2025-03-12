######    
blr="1e-3"
ema_decay_init="0.7"
feature_layer="11"
ema_decay_final='0.95'
warmup_epochs='20'
ema_decay_warmup_epoch="40"

CUDA_VISIBLE_DEVICES=0 python main_bootstrapped_pretrain.py \
--blr "$blr" \
--ema_decay_init "$ema_decay_init" \
--feature_layer "$feature_layer" \
--ema_decay_final "$ema_decay_final" \
--warmup_epochs "$warmup_epochs" \
--ema_decay_warmup_epoch "$ema_decay_warmup_epoch" \
--output_dir ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"_warmup_epochs"$warmup_epochs"_ema_decay_warmup_epoch"$ema_decay_warmup_epoch"_l22

CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"_warmup_epochs"$warmup_epochs"_ema_decay_warmup_epoch"$ema_decay_warmup_epoch"_l22/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_finetune/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"_warmup_epochs"$warmup_epochs"_ema_decay_warmup_epoch"$ema_decay_warmup_epoch"_l22

CUDA_VISIBLE_DEVICES=0 python main_linprobe.py \
--finetune ./output_dir/bootmae_pretrain/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"_warmup_epochs"$warmup_epochs"_ema_decay_warmup_epoch"$ema_decay_warmup_epoch"_l22/checkpoint-199.pth \
--output_dir ./output_dir/bootmae_linprobe/blr"$blr"_ema"$ema_decay_init"_featurelayer"$feature_layer"_ema_decay_final"$ema_decay_final"_warmup_epochs"$warmup_epochs"_ema_decay_warmup_epoch"$ema_decay_warmup_epoch"_l22
######