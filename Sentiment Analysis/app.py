import numpy as np
import twitter_credentials

import re


from numpy import zeros 
from numpy import asarray 
import pandas as pd
from tweepy import API 
from tweepy import OAuthHandler


from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation,Dropout,Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt



data=[]
tweets=[]
dataaw=[]
datee=[]
like=[]
lists=[]

# # # # TWITTER Credentials# # # #
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client





# # # # TWITTER AUTHENTICATER # # # #
class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY,twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN,twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth
    
    
class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets.
    """

    def clean_tweet1(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())




    
if __name__ == '__main__':

  
    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()
    api = twitter_client.get_twitter_client_api()
    n=10    
    tweets = api.user_timeline(screen_name="SetuAarogya", count=n)
    
    data=[i.text for i in tweets]
    datee=np.array([i.created_at for i in tweets])
    like=np.array([i.favorite_count for i in tweets])
    print(data)
    print("************************************************************************************************************************************************************************")
    for i in data:
        dataaw=tweet_analyzer.clean_tweet1(i)
        lists.append(dataaw)
        print(dataaw)

reviews=pd.read_csv("E:\imdbdataset.csv")
reviews.isnull().values.any()

#Data Preprocessing
TAG_RE=re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('',text)

def clean_tweet( tweet):
    tweet=remove_tags(tweet)
    tweet=re.sub(r"\s+[a-zA-Z]\s+",'',tweet)
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


x=[]
sentence=list(reviews['review'])
for sen in sentence:
    x.append(clean_tweet(sen))
    

y=reviews['sentiment']
y=np.array(list(map(lambda x: 1 if x=="positive" else 0,y)))

x_train=x
y_train=y
#Embeding layer

tokenizer=Tokenizer(num_words=5000)
tokenizer.fit_on_texts(x_train)

x_train=tokenizer.texts_to_sequences(x_train)

vocab_size=len(tokenizer.word_index)+1
x_train=pad_sequences(x_train,padding='post',maxlen=60)



embeddings_dictionary = dict()
glove_file=open('E:/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()


embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

#model for text classification
model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=60 , trainable=False)
model.add(embedding_layer)
model.add(LSTM(128))


model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=128, epochs=2, verbose=1, validation_split=0.2)
print("************************************************************************************************************************************************************************")
for i in range(0,n):
    x_test=data[i]
    tokenizer.fit_on_texts(x_test)
    x_test=tokenizer.texts_to_sequences(x_test)
    x_test=pad_sequences(x_test,padding='post',maxlen=60)
    h=model.predict(x_test)
    l=np.mean(h)
    print(data[i])
    if l>.5:
      print("POSITIVE REVIEW")
    elif l<.5:
      print("Negative Review")
    else:
     print("Neutral Sentiment")
    print(l)
time_like=pd.Series(data=like,index=datee)
time_like.plot(figsize=(16,4),label="likes",legend=True)
plt.xlabel("Date")
plt.ylabel("Likes")
plt.show()

