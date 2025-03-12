


# baseline
######
python main_pretrain.py \
--blr 5e-4 \
--output_dir ./output_dir/mae_pretrain/blr5e-4

python main_finetune.py \
--finetune ./output_dir/mae_pretrain/blr5e-4/checkpoint-199.pth \
--output_dir ./output_dir/mae_finetune/blr5e-4

python main_linprobe.py \
--finetune ./output_dir/mae_pretrain/blr5e-4/checkpoint-199.pth \
--output_dir ./output_dir/mae_linprobe/blr5e-4
######

######
python main_pretrain.py \
--blr 2e-4 \
--output_dir ./output_dir/mae_pretrain/blr2e-4

python main_finetune.py \
--finetune ./output_dir/mae_pretrain/blr2e-4/checkpoint-199.pth \
--output_dir ./output_dir/mae_finetune/blr2e-4

python main_linprobe.py \
--finetune ./output_dir/mae_pretrain/blr2e-4/checkpoint-199.pth \
--output_dir ./output_dir/mae_linprobe/blr2e-4
######

######
python main_pretrain.py \
--blr 1e-3 \
--output_dir ./output_dir/mae_pretrain/blr1e-3

python main_finetune.py \
--finetune ./output_dir/mae_pretrain/blr1e-3/checkpoint-199.pth \
--output_dir ./output_dir/mae_finetune/blr1e-3

python main_linprobe.py \
--finetune ./output_dir/mae_pretrain/blr1e-3/checkpoint-199.pth \
--output_dir ./output_dir/mae_linprobe/blr1e-3
######