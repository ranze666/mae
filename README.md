## Env 

```
conda create -n bootmae python=3.6
conda activate bootmae
pip install --upgrade pip
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install tensorboard
```

## Experiment

The scripts will automatically download the CIFAR10 dataset. If the download speed is too slow, please download it manually and put it in the following location:
```
./data/cifar-10-python.tar.gz
```

Pretrain, finetune and linearprobe for original MAE:
```
bash scripts/mae_pretrain.sh
bash scripts/mae_finetuning.sh
bash scripts/mae_linear_eval.sh
```

Pretrain, finetune and linearprobe for bootstrapped MAE:
```
bash scripts/bootmae_pretrain.sh
bash scripts/bootmae_finetuning.sh
bash scripts/bootmae_linear_eval.sh
```

The results and checkpoints will be saved to

```
./output_dir/bootmae_finetune/test_results.txt
./output_dir/bootmae_linprobe/test_results.txt
./output_dir/mae_finetune/test_results.txt
./output_dir/mae_linprobe/test_results.txt
```