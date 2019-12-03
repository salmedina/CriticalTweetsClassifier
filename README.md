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
```

### Partial Results

#### Earthquake experiment
##### Data Split

train:
 - 2015_Nepal_Earthquake_en
 - 2014_Chile_Earthquake_en
 - 2012_Costa_Rica_earthquake
 - 2012_Guatemala_earthquake
 
valid:
 - 2012_Italy_earthquakes

##### Results
|       Model       |    Acc   | Macro F1 | Epoch |                                                                Critical Metrics                                                                |
|:-----------------:|:--------:|:--------:|:-----:|:----------------------------------------------------------------------------------------------------------------------------------------------:|
|    Baseline-GloVe | 0.830357 | 0.644277 |   21  |   {'low': [0.8529403402545683, 0.9560429054473566, 0.901503635067761], 'high': [0.5999940000599994, 0.2857129251765468, 0.38705057718924907]}  |
|     Baseline-BERT | 0.803571 | 0.576585 |   11  |   {'low': [0.8349506456789848, 0.9450539065341687, 0.8865472182306581], 'high': [0.4444395062277086, 0.19047528345103118, 0.266622895408111]}  |
|   Multitask-GloVe | 0.812500 | 0.660015 |   19  | {'low': [0.8645824327266326, 0.9120869097946046, 0.8876496239059914], 'high': [0.49999687501953116, 0.38095056690206236, 0.43238101360925524]} |
|    Mulittask-BERT | 0.803571 | 0.521326 |   11  |  {'low': [0.8224291379166935, 0.9670319043605446, 0.8888383202946046], 'high': [0.3999920001599968, 0.09523764172551559, 0.15381391131490188]} |
| Adversarial-GloVe | 0.812500 | 0.644394 |   24  |   {'low': [0.8571419825081811, 0.9230759087077925, 0.8888380196583165], 'high': [0.4999964285969386, 0.3333317460393046, 0.3999497200352658]}  |
|  Adversarial-BERT | 0.839286 | 0.579963 |   6   |   {'low': [0.8348616193930097, 0.9999989011001087, 0.909949497702883], 'high': [0.9999666677777408, 0.1428564625882734, 0.2499760431292765]}   |                                        |
