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


