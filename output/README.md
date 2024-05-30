---
base_model: /data/zzy/Models/bert-base-chinese
tags:
- generated_from_trainer
datasets:
- ner_dataset
metrics:
- precision
- recall
- f1
- accuracy
model-index:
- name: output
  results:
  - task:
      name: Token Classification
      type: token-classification
    dataset:
      name: ner_dataset
      type: ner_dataset
      config: default
      split: validation
      args: default
    metrics:
    - name: Precision
      type: precision
      value: 0.9652217054263565
    - name: Recall
      type: recall
      value: 0.9740775046312522
    - name: F1
      type: f1
      value: 0.9696293850495585
    - name: Accuracy
      type: accuracy
      value: 0.9968802166653316
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# output

This model is a fine-tuned version of [/data/zzy/Models/bert-base-chinese](https://huggingface.co//data/zzy/Models/bert-base-chinese) on the ner_dataset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0182
- Precision: 0.9652
- Recall: 0.9741
- F1: 0.9696
- Accuracy: 0.9969

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 128
- eval_batch_size: 128
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|:--------:|
| 0.0126        | 1.0   | 1798 | 0.0148          | 0.9521    | 0.9631 | 0.9576 | 0.9957   |
| 0.0064        | 2.0   | 3596 | 0.0133          | 0.9606    | 0.9665 | 0.9635 | 0.9964   |
| 0.0036        | 3.0   | 5394 | 0.0142          | 0.9620    | 0.9723 | 0.9671 | 0.9967   |
| 0.0022        | 4.0   | 7192 | 0.0162          | 0.9642    | 0.9730 | 0.9685 | 0.9968   |
| 0.0013        | 5.0   | 8990 | 0.0182          | 0.9652    | 0.9741 | 0.9696 | 0.9969   |


### Framework versions

- Transformers 4.41.1
- Pytorch 2.0.1+cu117
- Datasets 2.18.0
- Tokenizers 0.19.1
