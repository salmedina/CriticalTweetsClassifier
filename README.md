# Critical Tweets Classifier

This project is a work on detecting tweets that require an action from first responders or the government during an emerging crisis, such as a natural disaster or an attack. We use a domain-adaptation technique between
different events, by adversarially training a classifier removing the importance on the event per se, allowing the trained model to focus on the criticality of the tweet.

### Data

- [BERT Embeddings of dataset (json)](https://drive.google.com/file/d/1KsenF_qhRsKk67NbSKz9ESYNqtClYngQ/view?usp=sharing)
- [BERT Embeddings of dataset (npy)](https://drive.google.com/file/d/1DqNa49IurhF7OFqMdPYC5acq-zbkGbtU/view?usp=sharing)

### Dependencies

```
pip install easydict
pip install pytorch-revgrad
pip install bert-embedding
pip install pyyaml
pip install pytorch-transformers
pip install umap-learn
```

### Partial Results

#### Earthquake experiment 

##### Results

- Best set of hyper-parameters

| Learning Rate | Weight Decay | Momentum |
|:-------------:|:------------:|:--------:|
|    0.086189   |   0.012114   | 0.687299 |

Mean results over all the splits

|       Model       | Epoch |     Acc    |     F1     | F1 - Non-Crit |  F1 - Crit   |
|:-----------------:|:-----:|:----------:|:----------:|:-------------:|--------------|
|  Baseline - GloVe |  16.5 |  0.599972  | 0.40485025 | 0.6590847522  | 0.1506160809 |
|   Baseline - BERT |  28.5 |  0.7785435 |  0.561208  | 0.8691996021  | 0.2532161462 |
|   MultiTask GloVe | 19.75 |  0.853181  | 0.55147875 | 0.9163932501  | 0.186563989  |
|    MultiTask BERT | 25.25 |  0.7836775 | 0.56843425 | 0.8676274441  | 0.2692408236 |
| Adversarial GloVe | 22.75 | 0.85279025 |  0.5538475 | 0.9162805248  | 0.1914148948 |
|  Adversarial BERT |  9.75 | 0.78671875 |   0.58261  | 0.8701950573  | 0.2950255033 |

