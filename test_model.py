from tracemalloc import stop
from nltk.corpus import stopwords
from flask import Flask, request, render_template, redirect
import nltk
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
countvec = CountVectorizer()
set(stopwords.words('indonesian'))


def predict(text):
    komentar = text
    
    model = pickle.load(open("model/my_classifier.pkl", "rb"))
    vek = pd.read_pickle("model/bow.pkl")

    stop_words = stopwords.words('indonesian')

    text = text.lower()

    finals = ''.join(c for c in text if not c.isdigit())
    stops = ' '.join(
        [word for word in finals.split() if word not in stop_words])
    
    # print(processed_doc1)
    # stopword = "Dengan hasil stopword = "+stops
    # cv = CountVectorizer()
    # data = [komentar]
    # vect = model.fit_transform([komentar]).toarray()
    tokens = nltk.word_tokenize(stops)
    # vector = vek.transform([stops])
    # v = vector.toarray()

    # text_array = np.array([stops])
    text_vector = countvec.fit_transform([stops])

    count_vect_df = pd.DataFrame(text_vector.todense(), columns=countvec.get_feature_names_out())
    v = pd.concat([vek, count_vect_df], axis=0)

    v = v.replace(np.nan, 0)
    print(v.head())
    print(vek.head())
    print(count_vect_df.head())
    v = v.iloc[:, 0 :4885]

    prediksi = model.predict(v) # (1 x 4885 )
    proba = model.predict_proba(v)
    if prediksi == 1:
        prediksi = "POSITIF"
        nega = "{:.0%}".format(proba[0][0])
        posi = "{:.0%}".format(proba[0][1])
    elif prediksi == 0:
        prediksi = "NEGATIF"
        nega = "{:.0%}".format(proba[0][0])
        posi = "{:.0%}".format(proba[0][1])
    else:
        print("Nothing")

    # return (render_template('index.html', variable=prediksi, neg=nega, pos=posi, final=finals, token=tokens, sw=stops))
    
    return (prediksi, nega, posi)

# print(predict('oke'))
