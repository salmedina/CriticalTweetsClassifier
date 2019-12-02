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
```

### Partial Results

#### Earthquake experiment

|       Model       | Acc      | Macro F1 | Epoch | Critical                                                                                                                                       |
|:-----------------:|----------|----------|-------|------------------------------------------------------------------------------------------------------------------------------------------------|
|  Baseline - GloVe | 0.794643 | 0.591256 | 19    | {'low': [0.83999916000084, 0.9230759087077925, 0.879530344651999], 'high': [0.4166631944733794, 0.238094104313789, 0.30298219252467534]}       |
|   Baseline - BERT | 0.785714 | 0.439975 | 19    | {'low': [0.8073387088635698, 0.9670319043605446, 0.8799495277949881], 'high': [0.0, 0.0, 0.0]}                                                 |
|   MultiTask GloVe | 0.812500 | 0.660015 | 19    | {'low': [0.8645824327266326, 0.9120869097946046, 0.8876496239059914], 'high': [0.49999687501953116, 0.38095056690206236, 0.43238101360925524]} |
|    MultiTask BERT | 0.767857 | 0.499610 | 19    | {'low': [0.8155331888027293, 0.9230759087077925, 0.8659286828524662], 'high': [0.2222197531138543, 0.09523764172551559, 0.1332904575696939]}   |
| Adversarial GloVe | 0.776786 | 0.576669 | 19    | {'low': [0.8367338400675101, 0.9010979108814166, 0.867674020958463], 'high': [0.35714030614067044, 0.238094104313789, 0.2856646611103962]}     |
|  Adversarial BERT | 0.794643 | 0.442761 | 19    | {'low': [0.8090901735543876, 0.9780209032737326, 0.8855217076809389], 'high': [0.0, 0.0, 0.0]}                                                 |
