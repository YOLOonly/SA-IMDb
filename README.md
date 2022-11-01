# SA-IMDb
Sentiment analysis on IMDB dataset.


## Usage
Because the dataset contains too many files, it is compressed here.

Unzip the dataset before starting

```bash
unzip aclImdb.zip
```

then run the following command to start training

```bash
nohup python -um model.bilstm > ./train.log 2>&1 &
```

## Performance


|      model      | Accuracy |
| :-------------: | :------: |
|     BiLSTM      |  0.7911  |
| BiLSTM w/ GloVe |  0.8761  |
|     TextCNN     |  0.8698  |
|      BERT       |  0.8532  |
