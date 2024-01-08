# DP-AdamBC

This is the code repository for [DP-AdamBC: Your DP-Adam Is Actually DP-SGD (Unless You Apply Bias Correction)](https://arxiv.org/abs/2312.14334).

### Bash script examples
#### Jax-implementation

```python
# DP-AdamBC
python main.py --seed 1024 --epochs 70 --progress_bar --batch_size 1024 \
    --learning_rate 0.000005 --dataset CIFAR10 --classifier_model CNN5 \
    --activation tanh --dp_l2_norm_clip 3.0 --trainer DPAdam --adam_corr \
    --eps_root 0.000000001 --target_eps 1.0 --debug

# DP-Adam
python main.py --seed 1024 --epochs 70 --progress_bar --batch_size 512 \
    --learning_rate 0.001 --dataset CIFAR10 --classifier_model CNN5 \
    --activation tanh --dp_l2_norm_clip 3.0 --trainer DPAdam --target_eps 1.0 --debug

# DP-SGD
python main.py --seed 1024 --epochs 70 --progress_bar --batch_size 1024 \
    --learning_rate 0.01 --dataset CIFAR10 --classifier_model CNN5 \
    --activation tanh --dp_l2_norm_clip 3.0 --trainer DPIterative --target_eps 1.0 --debug
```

(DP-Adam and DP-SGD-Momentum comparison example)

```python
python main.py --seed 1024 --data_dir /home/qiaoyuet/data --epochs 30 --progress_bar \
    --batch_size 256 --dp_noise_multiplier 1.0 --dp_l2_norm_clip 1.0 --learning_rate 0.0256 \
    --dataset CIFAR10 --classifier_model CNN5 --activation tanh --trainer DPAdam --sgd_momentum 
    --beta_1 0.9 --debug

python main.py --seed 1024 --data_dir /home/qiaoyuet/data --epochs 30 --progress_bar \
    --batch_size 256 --dp_noise_multiplier 1.0 --dp_l2_norm_clip 1.0 --learning_rate 0.001 \
    --dataset CIFAR10 --classifier_model CNN5 --activation tanh --trainer DPAdam \
    --beta_1 0.9 --eps 1e-8 --debug
```

#### Pytorch-implementation

```python
# DP-AdamBC
python text_classification_glue.py --opt_model adam_corr --seed 1024 --batch_size 256 \
    --train_from_scratch --lr 0.003 --num_epochs 10 --eps_root 0.000000007 --data_name qnli \
    --model_name bert_base --target_epsilon 3.0

# DP-Adam
python text_classification_glue.py --opt_model adam --seed 1024 --batch_size 256 \
    --train_from_scratch --lr 0.01 --num_epochs 7 --data_name qnli --model_name bert_base \
    --target_epsilon 3.0

# DP-SGD
python text_classification_glue.py --opt_model sgd --seed 1024 --batch_size 256 \
    --train_from_scratch --lr 30 --num_epochs 5 --data_name qnli --model_name bert_base \
    --target_epsilon 3.0
```

(DP-Adam and DP-SGD-Momentum comparison example)

```python
python opacus_text_classification.py --opt_model adam --batch_size 256 --train_from_scratch \
    --lr 0.01 --num_epochs 5 --seed 1024 --dp_noise_multiplier 0.4 --dp_l2_norm_clip 0.1 

python opacus_text_classification.py --opt_model sgdm --batch_size 256 --train_from_scratch \
    --lr 6.4 --num_epochs 5 --seed 1024 --dp_noise_multiplier 0.4 --dp_l2_norm_clip 0.1
```