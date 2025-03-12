
######
python main_bootstrapped_pretrain.py \
--output_dir ./output_dir/final/bootmae_pretrain

python main_finetune.py \
--finetune ./output_dir/final/bootmae_pretrain/checkpoint-199.pth \
--output_dir ./output_dir/final/bootmae_finetune

python main_linprobe.py \
--finetune ./output_dir/final/bootmae_pretrain/checkpoint-199.pth \
--output_dir ./output_dir/final/bootmae_linprobe
######

######
python main_pretrain.py \
--blr 1e-3 \
--output_dir ./output_dir/final/mae_pretrain

python main_finetune.py \
--finetune ./output_dir/final/mae_pretrain/checkpoint-199.pth \
--output_dir ./output_dir/final/mae_finetune

python main_linprobe.py \
--finetune ./output_dir/final/mae_pretrain/checkpoint-199.pth \
--output_dir ./output_dir/final/mae_linprobe
######

python main_finetune.py \
--output_dir ./output_dir/final/scratch_finetune

python main_linprobe.py \
--output_dir ./output_dir/final/scratch_linprobe