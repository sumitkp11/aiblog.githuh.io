---
title: Fine-tune a Pre-trained model for Custom Domain
description: Let’s see the following points covered in the course provided by Nvidia's Deep Learning Institue on Transformers-based Natural Language Processing
date: 2026-01-21
tags: ["nvidia", "transformers", "model", "token"]
---

# Fine-tune a pre-trained model for custom domain

An NER model is typically comprised of a pre-trained BERT model followed by a token classification layer.

For training, the config file consists of sections such as:
- **model**: language model, datasets, token classifier, optimizer and schedulers.
- **trainer**: any argument that is to be passed to PyTorch Lightning.

For beginners, NeMo provides a starter configuration file.

## Configuration File
```
# define config path
MODEL_CONFIG = "token_classification_config.yaml"
WORK_DIR = "WORK_DIR"
os.makedirs(WORK_DIR, exist_ok=True)

# download the model's configuration file 
BRANCH = 'main'
config_dir = WORK_DIR + '/configs/'
os.makedirs(config_dir, exist_ok=True)

if not os.path.exists(config_dir + MODEL_CONFIG):
    print('Downloading config file...')
    wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/examples/nlp/token_classification/conf/' + MODEL_CONFIG, config_dir)
else:
    print ('config file already exists')
```

Here, the config file for NER, `token_classification_config.yaml`, specifies model, training and experiment management details such as file locations, pretrained models and hyperparameters.


```python
from omegaconf import OmegaConf

CONFIG_DIR = "/dli/task/WORK_DIR/configs"
CONFIG_FILE = "token_classification_config.yaml"

config=OmegaConf.load(CONFIG_DIR + "/" + CONFIG_FILE)

# print the entire configuration file
print(OmegaConf.to_yaml(config))
```

#### output
```bash
pretrained_model: null
trainer:
  devices: 1
  num_nodes: 1
  max_epochs: 5
  max_steps: -1
  accumulate_grad_batches: 1
  gradient_clip_val: 0.0
  precision: 16
  accelerator: gpu
  enable_checkpointing: false
  logger: false
  log_every_n_steps: 1
  val_check_interval: 1.0
exp_manager:
  exp_dir: null
  name: token_classification_model
  create_tensorboard_logger: true
  create_checkpoint_callback: true
model:
  label_ids: null
  class_labels:
    class_labels_file: label_ids.csv
  dataset:
    data_dir: ???
    class_balancing: null
    max_seq_length: 128
    pad_label: O
    ignore_extra_tokens: false
    ignore_start_end: false
    use_cache: false
    num_workers: 2
    pin_memory: false
    drop_last: false
  train_ds:
    text_file: text_train.txt
    labels_file: labels_train.txt
    shuffle: true
    num_samples: -1
    batch_size: 64
  validation_ds:
    text_file: text_dev.txt
    labels_file: labels_dev.txt
    shuffle: false
    num_samples: -1
    batch_size: 64
  test_ds:
    text_file: text_dev.txt
    labels_file: labels_dev.txt
    shuffle: false
    num_samples: -1
    batch_size: 64
  tokenizer:
    tokenizer_name: ${model.language_model.pretrained_model_name}
    vocab_file: null
    tokenizer_model: null
    special_tokens: null
  language_model:
    pretrained_model_name: bert-base-uncased
    lm_checkpoint: null
    config_file: null
    config: null
  head:
    num_fc_layers: 2
    fc_dropout: 0.5
    activation: relu
    use_transformer_init: true
  optim:
    name: adam
    lr: 5.0e-05
    weight_decay: 0.0
    sched:
      name: WarmupAnnealing
      warmup_steps: null
      warmup_ratio: 0.1
      last_epoch: -1
      monitor: val_loss
      reduce_on_plateau: false
hydra:
  run:
    dir: .
  job_logging:
    root:
      handlers: null
```
```python
# in this exercise, train and dev datasets are located in the same folder under the default names, 
# so it is enough to add the path of the data directory to the config
config.model.dataset.data_dir = os.path.join(DATA_DIR, 'NER')

# print the model section
print(OmegaConf.to_yaml(config.model))
```

#### output
```bash
label_ids: null
class_labels:
  class_labels_file: label_ids.csv
dataset:
  data_dir: data/NCBI/NER
  class_balancing: null
  max_seq_length: 128
  pad_label: O
  ignore_extra_tokens: false
  ignore_start_end: false
  use_cache: false
  num_workers: 2
  pin_memory: false
  drop_last: false
train_ds:
  text_file: text_train.txt
  labels_file: labels_train.txt
  shuffle: true
  num_samples: -1
  batch_size: 64
validation_ds:
  text_file: text_dev.txt
  labels_file: labels_dev.txt
  shuffle: false
  num_samples: -1
  batch_size: 64
test_ds:
  text_file: text_dev.txt
  labels_file: labels_dev.txt
  shuffle: false
  num_samples: -1
  batch_size: 64
tokenizer:
  tokenizer_name: ${model.language_model.pretrained_model_name}
  vocab_file: null
  tokenizer_model: null
  special_tokens: null
language_model:
  pretrained_model_name: bert-base-uncased
  lm_checkpoint: null
  config_file: null
  config: null
head:
  num_fc_layers: 2
  fc_dropout: 0.5
  activation: relu
  use_transformer_init: true
optim:
  name: adam
  lr: 5.0e-05
  weight_decay: 0.0
  sched:
    name: WarmupAnnealing
    warmup_steps: null
    warmup_ratio: 0.1
    last_epoch: -1
    monitor: val_loss
    reduce_on_plateau: false
```

## Download domain-specific pre-trained models



