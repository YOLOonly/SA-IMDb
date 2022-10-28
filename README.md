# SA-IMDb
Sentiment analysis on IMDB dataset.


## Usage
Because the dataset contains too many files, the method of compressing and uploading is adopted.


Before you start, you need to unzip the dataset

```bash
unzip aclImdb.zip
```

then run the following command to start training

```bash
nohup python -um model.bilstm > ./train.log 2>&1 &
```

## Performance


|model | Accuracy |
| :-: | :-: |
|bilstm| 0.7836|
