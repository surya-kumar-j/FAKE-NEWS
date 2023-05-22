from flask import Flask,redirect,render_template,request,flash,url_for
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from keras.models import load_model
import nltk
import re
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split

app=Flask(__name__)
### Vocabulary size
voc_size=5000

## Creating model
sent_length=20
embedding_vector_features=40



model=load_model("new_model.h5")


#model=pickle.load(open(".pickle","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/fakenews.html')
def fake():
    return render_template('fakenews.html')

@app.route('/fakenews',methods=['POST'])
def fakenews():
    input_from_user = request.form['fakenews']
    ps = PorterStemmer()
    input_data = []

    print("Applying regex")
    input_sample = re.sub('[^a-zA-Z]', ' ', input_from_user)
    input_sample = input_sample.lower()
    input_sample = input_sample.split()
    print("detecting stop words") 
    input_sample = [ps.stem(word) for word in input_sample if not word in stopwords.words('english')]
    input_sample = ' '.join(input_sample)
    input_data.append(input_sample)
    print(input_sample)

    onehot_repr_input_data=[one_hot(words,voc_size)for words in input_data] 
    sent_length=20
    embedded_docs_input_data=pad_sequences(onehot_repr_input_data,padding='pre',maxlen=sent_length)
    X_final_input_data=np.array(embedded_docs_input_data)
    y_pred_input_data=model.predict(X_final_input_data)
    print(y_pred_input_data)
    
    
    outpred=np.argmax(y_pred_input_data[0])
    print(outpred)
    output="fake news" if outpred==1 else "True news"
    print(output)
    
    
    
    return render_template('fakenews.html',s1=output)


if __name__=='__main__':

    
    app.run(debug=True)
