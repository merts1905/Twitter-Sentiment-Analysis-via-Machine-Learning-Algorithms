# Twitter-Sentiment-Analysis-via-Machine-Learning-Algorithms
Twitter sentiment analysis on live tweets using 5 different machine learning algorithms : SVM,NB,RF,CNN,LSTM

** YOU NEED YOUR OWN API KEY FOR THIS PROJECT ! **

# Requirements

-Python 3.x
-Anaconda
-Jupyter Notebook
-Nvidia cudNN
-tensorflow
-nltk
-pandas
-keras 2.3.1
-sklearn
-numpy
-twitter API
-regex
-sentiment140 dataset from kaggle
-GloVe word embedding


# How to use

-I used sentiment140 dataset on kaggle for training algorithms and GloVe word embedding

-This dataset has 800k negative labeled tweets and 200k positive labeled tweets and mentions

-I shuffled data randomly using shuffle.py and using main.py i divide dataset into 5 different size so that i can see the effects of dataset size on my project.These divided datasets have %60 negative labeled tweets and %40 positive labeled tweets

-Train models with LSTN.py,CNN.py,NB.py,RF.py,SVM.py

-It generates pickle file that saves model 

-Using this saved model you can make predictions on live tweets

-Open jupyter notebook and run the algorithm which you prefer on prediction folder

-Write your own API keys

-Write your own search keyword (q=??)




