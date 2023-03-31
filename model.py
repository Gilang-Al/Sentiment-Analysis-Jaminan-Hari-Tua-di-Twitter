import numpy as np
import pandas as pd
import json, nltk
import matplotlib.pyplot as plt
# from wordcloud import WordCloud
import seaborn as sns
import pickle
# nltk.download('wordnet')   # for Lemmatization
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# %matplotlib inline

def all_in_modeling(filename):
    data = pd.read_csv(filename)
    from sklearn.feature_extraction.text import CountVectorizer
    countvec = CountVectorizer()
    cdf = countvec.fit_transform(data["text_normal"])

    bow = pd.DataFrame(cdf.toarray(), columns = countvec.get_feature_names_out())

    data.info()

    # split data
    X_train, X_test, y_train, y_test = train_test_split(bow, data["polarity"],test_size=0.2, random_state=69) 

    print("X_train_shape : ",X_train.shape)
    print("X_test_shape : ",X_test.shape)
    print("y_train_shape : ",y_train.shape)
    print("y_test_shape : ",y_test.shape)

    # Naive Bayes Classifier
    model_naive = MultinomialNB().fit(X_train, y_train) 
    predicted_naive = model_naive.predict(X_test)


    # cofusion matrix
    plt.figure(dpi=200)
    mat = confusion_matrix(y_test, predicted_naive)
    sns.heatmap(mat.T, annot=True, fmt='d', cbar=False)

    plt.title('Confusion Matrix for Naive Bayes')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig("confusion_matrix.png")
    # plt.show()

    # menampilkan akurasi
    score_naive = accuracy_score(predicted_naive, y_test)
    print("Accuracy with Naive-bayes testing: ",score_naive)
    print(classification_report(y_test, predicted_naive))

    data_sentimen = pd.read_csv(filename)

    def TestModel (data):
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB 
        data_sentimen = pd.read_csv(filename)
        data = pd.read_csv(filename)
        countvec = CountVectorizer() #bow
        cdf = countvec.fit_transform(data_sentimen["text_normal"])
        bow = pd.DataFrame(cdf.toarray(), columns = countvec.get_feature_names()) 
        X_train, X_test, y_train, y_test = train_test_split(bow, data["polarity"],test_size=0.1, random_state=69) 
        model_naive = MultinomialNB().fit(X_train, y_train) 
        std_pred = model_naive.predict(bow)
        df = pd.DataFrame(data = {'Text' : data_sentimen['text_normal'], 'Polarity' : data_sentimen['polarity'], 'Sentimen' : std_pred})
        return df

    TestModel(data_sentimen['text_normal'])

    TestModel(data_sentimen['text_normal']).to_csv("data_sentimen.csv")

    data_sentimen = pd.read_csv('data_sentimen.csv')

    data_sentimen['Sentimen'].value_counts()

    def polarity_encode(x):
        if(x == 1):
            return 'Positif'
        if(x == 0):
            return 'Negatif'
    # data_sentimen.Sentimen = data_sentimen.Sentimen.apply(polarity_encode)

    fig, ax = plt.subplots(figsize = (6, 6))
    sizes = [count for count in data_sentimen['Sentimen'].value_counts()]
    labels = list(data_sentimen['Sentimen'].value_counts().index)
    explode = (0, 0)
    ax.pie(x = sizes, labels = labels, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
    fig.canvas.manager.set_window_title('Hasil Klasifikasi Sentimen Model')
    plt.title('Hasil Klasifikasi Sentimen Model')
    plt.show()