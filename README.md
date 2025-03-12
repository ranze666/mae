## Env 

```
conda create -n bootmae python=3.6
conda activate bootmae
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install tensorboard
```

## Experiment
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
