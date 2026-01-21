---
title: Transformers-based Natural Language Processing 
description: Let’s see the following points covered in the course provided by Nvidia's Deep Learning Institue on Transformers-based Natural Language Processing
date: 2026-01-21
tags: ["nvidia", "transformers", "model", "token"]
---

## Download and Preprocess Data

```
import os
import wget

# set data path
DATA_DIR="data/GMB"

# check that data folder should contain 4 files
!ls -l $DATA_DIR
```

### output
```
total 11140
-rw-r--r-- 1 root root      77 Jan 21 16:38 label_ids.csv
-rw-r--r-- 1 root root  407442 Jan 21 16:38 labels_dev.txt
-rw-r--r-- 1 root root 3169783 Jan 21 16:38 labels_train.txt
-rw-r--r-- 1 root root  891020 Jan 21 16:38 text_dev.txt
-rw-r--r-- 1 root root 6928251 Jan 21 16:38 text_train.txt
```

```
# preview data 
print('Text:')
!head -n 5 {DATA_DIR}/text_train.txt

print('Labels:')
!head -n 5 {DATA_DIR}/labels_train.txt
```
### output
```
Text:
New Zealand 's cricket team has scored a morale-boosting win over Bangladesh in the first of three one-day internationals in New Zealand .
Despite Bangladesh 's highest total ever in a limited-overs match , the Kiwis were able to win the match by six wickets in Auckland .
Opening batsman Jamie How led all scorers with 88 runs as New Zealand reached 203-4 in 42.1 overs .
The score was in response to Bangladesh 's total of 201 all out in 46.3 overs .
Mohammad Ashraful led the visitors with 70 runs , including 10 fours and one six on the short boundaries of the Eden Park ground .
Labels:
B-LOC I-LOC O O O O O O O O O B-LOC O O B-TIME I-TIME I-TIME I-TIME O O B-LOC I-LOC O
O B-LOC O O O O O O O O O O B-GPE O O O O O O O O O O B-LOC O
O O B-PER I-PER O O O O O O O B-LOC I-LOC O O O O O O
O O O O O O B-LOC O O O O O O O O O O
B-PER I-PER O O O O O O O O O O O O O O O O O O O B-LOC I-LOC O O
```

### Download Pre-trained model

```
# import dependencies
from nemo.collections.nlp.models import TokenClassificationModel

# list available pre-trained models
for model in TokenClassificationModel.list_available_models():
    print(model)
```

### output
```
NOTE! Installing ujson may make loading annotations faster.
PretrainedModelInfo(
	pretrained_model_name=ner_en_bert,
	description=The model was trained on GMB (Groningen Meaning Bank) corpus for entity recognition and achieves 74.61 F1 Macro score.,
	location=https://api.ngc.nvidia.com/v2/models/nvidia/nemo/ner_en_bert/versions/1.10/files/ner_en_bert.nemo
)
```

```
# download and load the pre-trained BERT-based model
pretrained_ner_model=TokenClassificationModel.from_pretrained("ner_en_bert")
```

### output
```
# download and load the pre-trained BERT-based model
pretrained_ner_model=TokenClassificationModel.from_pretrained("ner_en_bert")
# download and load the pre-trained BERT-based model
pretrained_ner_model=TokenClassificationModel.from_pretrained("ner_en_bert")
[NeMo I 2026-01-21 17:08:48 cloud:68] Downloading from: https://api.ngc.nvidia.com/v2/models/nvidia/nemo/ner_en_bert/versions/1.10/files/ner_en_bert.nemo to /root/.cache/torch/NeMo/NeMo_1.20.0/ner_en_bert/8186f86c83b11d70b43b9ead695e7eda/ner_en_bert.nemo
[NeMo I 2026-01-21 17:08:49 common:913] Instantiating model from pre-trained checkpoint
[NeMo I 2026-01-21 17:08:52 tokenizer_utils:130] Getting HuggingFace AutoTokenizer with pretrained_model_name: bert-base-uncased, vocab_file: /tmp/tmp9g1j_hsl/tokenizer.vocab_file, merges_files: None, special_tokens_dict: {}, and use_fast: False
Downloading tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]
Downloading config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]
Downloading vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]
Using eos_token, but it is not set yet.
Using bos_token, but it is not set yet.
[NeMo W 2026-01-21 17:08:53 modelPT:244] You tried to register an artifact under config key=tokenizer.vocab_file but an artifact for it has already been registered.
[NeMo W 2026-01-21 17:08:53 modelPT:161] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
    Train config : 
    text_file: text_train.txt
    labels_file: labels_train.txt
    shuffle: true
    num_samples: -1
    batch_size: 64
    
[NeMo W 2026-01-21 17:08:53 modelPT:168] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s). 
    Validation config : 
    text_file: text_dev.txt
    labels_file: labels_dev.txt
    shuffle: false
    num_samples: -1
    batch_size: 64
    
[NeMo W 2026-01-21 17:08:53 modelPT:174] Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method and provide a valid configuration file to setup the test data loader(s).
    Test config : 
    text_file: text_dev.txt
    labels_file: labels_dev.txt
    shuffle: false
    num_samples: -1
    batch_size: 64
    
Downloading model.safetensors:   0%|          | 0.00/440M [00:00<?, ?B/s]
[NeMo W 2026-01-21 17:08:56 modelPT:244] You tried to register an artifact under config key=language_model.config_file but an artifact for it has already been registered.
[NeMo I 2026-01-21 17:08:57 save_restore_connector:249] Model TokenClassificationModel was successfully restored from /root/.cache/torch/NeMo/NeMo_1.20.0/ner_en_bert/8186f86c83b11d70b43b9ead695e7eda/ner_en_bert.nemo.
```


### Make Predictions
```
# define the list of queries for inference
queries=[
    'we bought four shirts from the nvidia gear store in santa clara.',
    'Nvidia is a company.',
]

# make sample predictions
results=pretrained_ner_model.add_predictions(queries)

# show predictions
for query, result in zip(queries, results):
    print(f'Query : {query}')
    print(f'Result: {result.strip()}\n')
    print()
```
### output
```
# define the list of queries for inference
queries=[
    'we bought four shirts from the nvidia gear store in santa clara.',
    'Nvidia is a company.',
]

# make sample predictions
results=pretrained_ner_model.add_predictions(queries)

# show predictions
for query, result in zip(queries, results):
    print(f'Query : {query}')
    print(f'Result: {result.strip()}\n')
    print()
# define the list of queries for inference
queries=[
    'we bought four shirts from the nvidia gear store in santa clara.',
    'Nvidia is a company.',
]
​
# make sample predictions
results=pretrained_ner_model.add_predictions(queries)
​
# show predictions
for query, result in zip(queries, results):
    print(f'Query : {query}')
    print(f'Result: {result.strip()}\n')
    print()
[NeMo I 2026-01-21 17:10:06 token_classification_dataset:123] Setting Max Seq length to: 17
[NeMo I 2026-01-21 17:10:06 data_preprocessing:404] Some stats of the lengths of the sequences:
[NeMo I 2026-01-21 17:10:06 data_preprocessing:406] Min: 9 |                  Max: 17 |                  Mean: 13.0 |                  Median: 13.0
[NeMo I 2026-01-21 17:10:06 data_preprocessing:412] 75 percentile: 15.00
[NeMo I 2026-01-21 17:10:06 data_preprocessing:413] 99 percentile: 16.92
[NeMo W 2026-01-21 17:10:06 token_classification_dataset:152] 0 are longer than 17
[NeMo I 2026-01-21 17:10:06 token_classification_dataset:155] *** Example ***
[NeMo I 2026-01-21 17:10:06 token_classification_dataset:156] i: 0
[NeMo I 2026-01-21 17:10:06 token_classification_dataset:157] subtokens: [CLS] we bought four shirts from the n ##vid ##ia gear store in santa clara . [SEP]
[NeMo I 2026-01-21 17:10:06 token_classification_dataset:158] loss_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
[NeMo I 2026-01-21 17:10:06 token_classification_dataset:159] input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
[NeMo I 2026-01-21 17:10:06 token_classification_dataset:160] subtokens_mask: 0 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 0
Query : we bought four shirts from the nvidia gear store in santa clara.
Result: we bought four shirts from the nvidia[B-ORG] gear store in santa[B-LOC] clara[I-LOC].


Query : Nvidia is a company.
Result: Nvidia[B-ORG] is a company.
```

### Evaluate Predictions
```
# create a subset of our dev data
!head -n 100 $DATA_DIR/text_dev.txt > $DATA_DIR/sample_text_dev.txt
!head -n 100 $DATA_DIR/labels_dev.txt > $DATA_DIR/sample_labels_dev.txt

WORK_DIR = "WORK_DIR"

# evaluate model performance on sample
pretrained_ner_model.evaluate_from_file(
    text_file=os.path.join(DATA_DIR, 'sample_text_dev.txt'),
    labels_file=os.path.join(DATA_DIR, 'sample_labels_dev.txt'),
    output_dir=WORK_DIR,
    add_confusion_matrix=True,
    normalize_confusion_matrix=True,
    batch_size=1
)
```

#### output
```
[NeMo I 2026-01-21 17:12:02 token_classification_dataset:123] Setting Max Seq length to: 70
[NeMo I 2026-01-21 17:12:02 data_preprocessing:404] Some stats of the lengths of the sequences:
[NeMo I 2026-01-21 17:12:02 data_preprocessing:406] Min: 11 |                  Max: 70 |                  Mean: 26.9 |                  Median: 26.0
[NeMo I 2026-01-21 17:12:02 data_preprocessing:412] 75 percentile: 33.00
[NeMo I 2026-01-21 17:12:02 data_preprocessing:413] 99 percentile: 65.05
[NeMo W 2026-01-21 17:12:02 token_classification_dataset:152] 0 are longer than 70
[NeMo I 2026-01-21 17:12:02 token_classification_dataset:155] *** Example ***
[NeMo I 2026-01-21 17:12:02 token_classification_dataset:156] i: 0
[NeMo I 2026-01-21 17:12:02 token_classification_dataset:157] subtokens: [CLS] hamas refuses to recognize israel , and has vowed to undermine palestinian leader mahmoud abbas ' s efforts to make peace with the jewish state . [SEP]
[NeMo I 2026-01-21 17:12:02 token_classification_dataset:158] loss_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[NeMo I 2026-01-21 17:12:02 token_classification_dataset:159] input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[NeMo I 2026-01-21 17:12:02 token_classification_dataset:160] subtokens_mask: 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[NeMo I 2026-01-21 17:12:03 token_classification_model:464] Labels save to /dli/task/WORK_DIR/infer_sample_text_dev.txt
[NeMo I 2026-01-21 17:12:03 token_classification_model:470] Predictions saved to /dli/task/WORK_DIR/infer_sample_text_dev.txt
[NeMo I 2026-01-21 17:12:03 utils_funcs:109] Confusion matrix saved to /dli/task/WORK_DIR/Normalized_Confusion_matrix_20260121-171203
[NeMo I 2026-01-21 17:12:03 token_classification_model:481]                        precision    recall  f1-score   support
    
          O (label id: 0)     0.9878    0.9895    0.9887      1805
      B-GPE (label id: 1)     0.9429    1.0000    0.9706        33
      B-LOC (label id: 2)     0.9103    0.9103    0.9103        78
     B-MISC (label id: 3)     0.6667    1.0000    0.8000         2
      B-ORG (label id: 4)     0.8431    0.7544    0.7963        57
      B-PER (label id: 5)     0.8095    0.8644    0.8361        59
     B-TIME (label id: 6)     0.8936    0.9130    0.9032        46
      I-GPE (label id: 7)     1.0000    1.0000    1.0000         4
      I-LOC (label id: 8)     0.8000    0.8889    0.8421         9
     I-ORG (label id: 10)     0.8421    0.6809    0.7529        47
     I-PER (label id: 11)     0.8305    0.8750    0.8522        56
    I-TIME (label id: 12)     0.8462    0.8462    0.8462        13
    
                 accuracy                         0.9651      2209
                macro avg     0.8644    0.8935    0.8749      2209
             weighted avg     0.9650    0.9651    0.9647      2209
```
## Fine-tune a Pre-trained model
- Without specifying any config file, Nemo will use the default configurations for the model and trainer.
- When fine-tuning a pre-trained NER model, we need to setup training and evaluation data before training.

```
import pytorch_lightning as pl

# setup the data dir to get class weights statistics
pretrained_ner_model.update_data_dir(DATA_DIR)

# setup train and validation Pytorch DataLoaders
pretrained_ner_model.setup_training_data()
pretrained_ner_model.setup_validation_data()
```

#### output
```
[NeMo I 2026-01-21 17:18:54 token_classification_model:84] Setting model.dataset.data_dir to data/GMB.
[NeMo I 2026-01-21 17:18:54 token_classification_utils:118] Processing data/GMB/labels_train.txt
[NeMo I 2026-01-21 17:18:54 token_classification_utils:138] Using provided labels mapping {'O': 0, 'B-GPE': 1, 'B-LOC': 2, 'B-MISC': 3, 'B-ORG': 4, 'B-PER': 5, 'B-TIME': 6, 'I-GPE': 7, 'I-LOC': 8, 'I-MISC': 9, 'I-ORG': 10, 'I-PER': 11, 'I-TIME': 12}
[NeMo I 2026-01-21 17:18:54 token_classification_utils:154] Labels mapping {'O': 0, 'B-GPE': 1, 'B-LOC': 2, 'B-MISC': 3, 'B-ORG': 4, 'B-PER': 5, 'B-TIME': 6, 'I-GPE': 7, 'I-LOC': 8, 'I-MISC': 9, 'I-ORG': 10, 'I-PER': 11, 'I-TIME': 12} saved to : /dli/task/data/GMB/label_ids.csv
[NeMo I 2026-01-21 17:19:09 token_classification_utils:163] Three most popular labels in data/GMB/labels_train.txt:
[NeMo I 2026-01-21 17:19:09 data_preprocessing:194] label: 0, 1014899 out of 1199472 (84.61%).
[NeMo I 2026-01-21 17:19:09 data_preprocessing:194] label: 2, 43529 out of 1199472 (3.63%).
[NeMo I 2026-01-21 17:19:09 data_preprocessing:194] label: 6, 23321 out of 1199472 (1.94%).
[NeMo I 2026-01-21 17:19:09 token_classification_utils:165] Total labels: 1199472. Label frequencies - {0: 1014899, 2: 43529, 6: 23321, 4: 23215, 11: 19583, 10: 19515, 5: 19407, 1: 18074, 8: 8482, 12: 7555, 3: 1002, 9: 669, 7: 221}
[NeMo I 2026-01-21 17:19:09 token_classification_utils:171] Class weights restored from data/GMB/labels_train_weights.p
[NeMo W 2026-01-21 17:19:09 modelPT:244] You tried to register an artifact under config key=class_labels.class_labels_file but an artifact for it has already been registered.
[NeMo I 2026-01-21 17:19:12 token_classification_dataset:287] features restored from data/GMB/cached__text_train.txt__labels_train.txt__BertTokenizer_128_30522_-1
[NeMo I 2026-01-21 17:19:12 token_classification_utils:118] Processing data/GMB/labels_dev.txt
[NeMo I 2026-01-21 17:19:12 token_classification_utils:138] Using provided labels mapping {'O': 0, 'B-GPE': 1, 'B-LOC': 2, 'B-MISC': 3, 'B-ORG': 4, 'B-PER': 5, 'B-TIME': 6, 'I-GPE': 7, 'I-LOC': 8, 'I-MISC': 9, 'I-ORG': 10, 'I-PER': 11, 'I-TIME': 12}
[NeMo I 2026-01-21 17:19:12 token_classification_utils:160] data/GMB/labels_dev_label_stats.tsv found, skipping stats calculation.
[NeMo I 2026-01-21 17:19:12 token_classification_dataset:287] features restored from data/GMB/cached__text_dev.txt__labels_dev.txt__BertTokenizer_128_30522_-1
```

```
# set up loss
pretrained_ner_model.setup_loss()
```

#### output
```
CrossEntropyLoss()
```

```
# create a PyTorch Lightning trainer and call `fit` again
fast_dev_run=True
trainer=pl.Trainer(devices=1, accelerator='gpu', fast_dev_run=fast_dev_run)
trainer.fit(pretrained_ner_model)
```

#### output
```
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Running in `fast_dev_run` mode: will run the requested loop using 1 batch(es). Logging and checkpointing is suppressed.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[NeMo I 2026-01-21 17:18:51 modelPT:721] Optimizer config = Adam (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.999)
        capturable: False
        differentiable: False
        eps: 1e-08
        foreach: None
        fused: None
        lr: 5e-05
        maximize: False
        weight_decay: 0.0
    )
[NeMo I 2026-01-21 17:18:51 lr_scheduler:910] Scheduler "<nemo.core.optim.lr_scheduler.WarmupAnnealing object at 0x7fba462776d0>" 
    will be used during training (effective maximum steps = 1) - 
    Parameters : 
    (warmup_steps: null
    warmup_ratio: 0.1
    last_epoch: -1
    max_steps: 1
    )

  | Name                  | Type                 | Params
---------------------------------------------------------------
0 | bert_model            | BertEncoder          | 109 M 
1 | classifier            | TokenClassifier      | 600 K 
2 | loss                  | CrossEntropyLoss     | 0     
3 | classification_report | ClassificationReport | 0     
---------------------------------------------------------------
110 M     Trainable params
0         Non-trainable params
110 M     Total params
440.331   Total estimated model params size (MB)
[NeMo W 2026-01-21 17:18:51 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/connectors/data_connector.py:224: PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 16 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(
    
[NeMo W 2026-01-21 17:18:51 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/trainer.py:1609: PossibleUserWarning: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
      rank_zero_warn(
    
[NeMo W 2026-01-21 17:18:51 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/connectors/data_connector.py:224: PossibleUserWarning: The dataloader, val_dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 16 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(
    
Training: 0it [00:00, ?it/s]
Validation: 0it [00:00, ?it/s]
[NeMo I 2026-01-21 17:18:51 token_classification_model:159] 
    label                                                precision    recall       f1           support   
    O (label_id: 0)                                         98.89      99.48      99.18       1161
    B-GPE (label_id: 1)                                     90.91     100.00      95.24         20
    B-LOC (label_id: 2)                                     82.69      97.73      89.58         44
    B-MISC (label_id: 3)                                   100.00     100.00     100.00          2
    B-ORG (label_id: 4)                                     85.29      65.91      74.36         44
    B-PER (label_id: 5)                                     85.11      88.89      86.96         45
    B-TIME (label_id: 6)                                    95.83     100.00      97.87         23
    I-GPE (label_id: 7)                                    100.00     100.00     100.00          4
    I-LOC (label_id: 8)                                     66.67      80.00      72.73          5
    I-MISC (label_id: 9)                                     0.00       0.00       0.00          0
    I-ORG (label_id: 10)                                    86.36      63.33      73.08         30
    I-PER (label_id: 11)                                    92.11      85.37      88.61         41
    I-TIME (label_id: 12)                                  100.00     100.00     100.00          7
    -------------------
    micro avg                                               96.84      96.84      96.84       1426
    macro avg                                               90.32      90.06      89.80       1426
    weighted avg                                            96.81      96.84      96.72       1426
    
`Trainer.fit` stopped: `max_steps=1` reached.
```

```
# evaluate model performance on sample
pretrained_ner_model.evaluate_from_file(
    text_file=os.path.join(DATA_DIR, 'sample_text_dev.txt'),
    labels_file=os.path.join(DATA_DIR, 'sample_labels_dev.txt'),
    output_dir=WORK_DIR,
    add_confusion_matrix=True,
    normalize_confusion_matrix=True,
    batch_size=1
)
```
#### output
```
[NeMo I 2026-01-21 17:18:52 token_classification_dataset:123] Setting Max Seq length to: 70
[NeMo I 2026-01-21 17:18:52 data_preprocessing:404] Some stats of the lengths of the sequences:
[NeMo I 2026-01-21 17:18:52 data_preprocessing:406] Min: 11 |                  Max: 70 |                  Mean: 26.9 |                  Median: 26.0
[NeMo I 2026-01-21 17:18:52 data_preprocessing:412] 75 percentile: 33.00
[NeMo I 2026-01-21 17:18:52 data_preprocessing:413] 99 percentile: 65.05
[NeMo W 2026-01-21 17:18:52 token_classification_dataset:152] 0 are longer than 70
[NeMo I 2026-01-21 17:18:52 token_classification_dataset:155] *** Example ***
[NeMo I 2026-01-21 17:18:52 token_classification_dataset:156] i: 0
[NeMo I 2026-01-21 17:18:52 token_classification_dataset:157] subtokens: [CLS] hamas refuses to recognize israel , and has vowed to undermine palestinian leader mahmoud abbas ' s efforts to make peace with the jewish state . [SEP]
[NeMo I 2026-01-21 17:18:52 token_classification_dataset:158] loss_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[NeMo I 2026-01-21 17:18:52 token_classification_dataset:159] input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[NeMo I 2026-01-21 17:18:52 token_classification_dataset:160] subtokens_mask: 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[NeMo I 2026-01-21 17:18:53 token_classification_model:464] Labels save to /dli/task/WORK_DIR/infer_sample_text_dev.txt
[NeMo I 2026-01-21 17:18:53 token_classification_model:470] Predictions saved to /dli/task/WORK_DIR/infer_sample_text_dev.txt
[NeMo I 2026-01-21 17:18:53 utils_funcs:109] Confusion matrix saved to /dli/task/WORK_DIR/Normalized_Confusion_matrix_20260121-171853
[NeMo I 2026-01-21 17:18:53 token_classification_model:481]                        precision    recall  f1-score   support
    
          O (label id: 0)     0.9878    0.9895    0.9887      1805
      B-GPE (label id: 1)     0.9429    1.0000    0.9706        33
      B-LOC (label id: 2)     0.8556    0.9872    0.9167        78
     B-MISC (label id: 3)     1.0000    1.0000    1.0000         2
      B-ORG (label id: 4)     0.8636    0.6667    0.7525        57
      B-PER (label id: 5)     0.7794    0.8983    0.8346        59
     B-TIME (label id: 6)     0.9149    0.9348    0.9247        46
      I-GPE (label id: 7)     1.0000    1.0000    1.0000         4
      I-LOC (label id: 8)     0.7778    0.7778    0.7778         9
     I-ORG (label id: 10)     0.8611    0.6596    0.7470        47
     I-PER (label id: 11)     0.8868    0.8393    0.8624        56
    I-TIME (label id: 12)     0.8462    0.8462    0.8462        13
    
                 accuracy                         0.9651      2209
                macro avg     0.8930    0.8833    0.8851      2209
             weighted avg     0.9653    0.9651    0.9643      2209
    
```

```
# restart the kernel
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```




