def prepro(sk, nf):
    import pandas as pd
    import csv
    import numpy as np
    import datetime as dt
    import re
    import nltk
    import string
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from model import all_in_modeling

    from nltk import word_tokenize
    import numpy as np
    from string import punctuation
    import re
    import matplotlib.pyplot as plt

    data = pd.read_csv(nf)
    data_text = pd.DataFrame(data['tweet'])

    # mengecek spam
    data_text["tweet"].duplicated().sum(), len(data_text["tweet"])
    # menghilangkan spam
    data_text = data_text.drop_duplicates(subset=['tweet'], keep=False)
    data_text["tweet"].duplicated().sum(), len(data_text["tweet"])

    # cleaning
    #@title
    def text_cleaning(data):
        # hapus simbol
        data = re.sub('[^\w\s]', ' ', data)
        # hapus angka
        data = re.sub('\d+', '', data)
        # hapus extra whitespace
        data = ' '.join(data.split())
        # hapus emoji
        data = re.sub(r'[^\x00-\x7F]+', ' ', data)
        # hapus baris baru
        data = re.sub('\n', ' ', data)
        # hapus url link
        data = re.sub(r"http\S+", '', data)
        # hapus single char
        data = re.sub(r"\b[a-zA-Z]\b", "", data)
        # case folding
        data = data.lower()

        return data

    #@title
    # Kamus alay diambil dari ramaprakoso dan nasalsabila
    kamus_alay_1 = pd.read_csv(
        "./kamus alay/kbba.csv",
        usecols=["slang", "formal"])
    kamus_alay_2 = pd.read_csv(
        "./kamus alay/colloquial-indonesian-lexicon.csv",
        usecols=["slang", "formal"])

    kamus_alay = pd.concat([kamus_alay_1, kamus_alay_2])

    # Dictionary bahasa alay
    dict_alay = dict()
    for index, row in kamus_alay.iterrows():
        dict_alay[row['slang']] = row['formal']

    def normalize_text(data):
        word_tokens = word_tokenize(data)
        result = [dict_alay.get(w, w) for w in word_tokens]
        return ' '.join(result)

    # Tokenizing
    def tokenizingText(tweet): # Tokenizing or splitting a string, text into a list of tokens
        text = word_tokenize(tweet) 
        return text

    # stopword
    def filteringText(text): # Remove stopwors in a text
        listStopwords = set(stopwords.words('indonesian'))
        filtered = []
        for txt in text:
            if txt not in listStopwords:
                filtered.append(txt)
        text = filtered 
        return text

    #stemming
    def stemmingText(text): # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        text = [stemmer.stem(word) for word in text]
        return text

    # PREPROCESSING START
    data_text["text_cleaned"] = data_text["tweet"].apply(lambda x: text_cleaning(x))
    # data_text.head(5)

    # mencari data kosong
    data_text['text_cleaned'].replace('', np.nan, inplace=True)
    # data_text.head(5)

    # total data kosong
    data_text['text_cleaned'].isnull().sum()

    # hilangkan data kosong
    data_text.dropna(subset=['text_cleaned'], inplace=True)
    # data_text.head(5)

    # Total data setelah dihilangkan data kosong
    print(data_text.shape)
    data_text["text_cleaned"] = data_text["text_cleaned"].apply(lambda x: normalize_text(x.lower()))
    # data_text.head(10)

    # Memperbaiki ejaan secara manual
    dict_clean = {"hy" : "hi",
    "dl":"dulu"
    }
    def replace_word(data):
        word_tokens = word_tokenize(data)
        result = [dict_clean.get(w, w) for w in word_tokens]
        return ' '.join(result)

    data_text.insert(2, "text_normal", data_text["text_cleaned"].apply(lambda x: replace_word(x)))
    # data_text.head()

    data_text.reset_index(drop=True, inplace=True)
    # data_text

    # Tokenisasi data
    data_text['text_tokenize'] = data_text["text_normal"].apply(tokenizingText)
    # data_text.head(5)

    # Stemming
    data_text['text_stemming'] = data_text['text_tokenize'].apply(stemmingText)
    # data_text.head()

    # Filtering atau Stopword Removal
    data_text['text_filtering'] = data_text['text_stemming'].apply(filteringText)
    # data_text.head(10)

    # save data prepro
    data_text.to_csv("data_prepro.csv")

    # Lexicon

    # Determine sentiment polarity of tweets using indonesia sentiment lexicon (source : https://github.com/fajri91/InSet)
    # Loads lexicon positive and negative data
    lexicon_positive = dict()
    import csv
    with open('./Lexicon/positive.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            lexicon_positive[row[0]] = int(row[1])

    lexicon_negative = dict()
    import csv
    with open('./Lexicon/negative.csv', 'r') as csvfile: 
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            lexicon_negative[row[0]] = int(row[1])
            
    # Function to determine sentiment polarity of tweets        
    def sentiment_analysis_lexicon_indonesia(text):
        #for word in text:
        score = 0
        for word in text:
            if (word in lexicon_positive):
                score = score + lexicon_positive[word]
        for word in text:
            if (word in lexicon_negative):
                score = score + lexicon_negative[word]
        polarity=''
        if (score >= 0):
            polarity = '1'
        else:
            polarity = '0'
        return score, polarity
    #Results from determine sentiment polarity of tweets

    results = data_text['text_tokenize'].apply(sentiment_analysis_lexicon_indonesia)
    results = list(zip(*results))
    data_text['polarity_score'] = results[0]
    data_text['polarity'] = results[1]

    data_text.to_csv("./dataset/dataset "+sk+".csv")

    # fig, ax = plt.subplots(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    fig = plt.figure(figsize = (10, 5))
    fig.canvas.manager.set_window_title('Polarity score')
    data_text['polarity'].value_counts().plot(kind='bar')
    plt.title('Polarity score')
    # plt.show()
    all_in_modeling("./dataset/dataset "+sk+".csv")