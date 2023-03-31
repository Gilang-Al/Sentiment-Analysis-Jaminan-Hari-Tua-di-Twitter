import tweepy

access_token = "1162621805023993856-SfFnIeceCHmvFBBCoTYwX7pod2pMAR"
access_token_secret = "i1KvuP3ECB88i94G5ReDkzIM6CfJlM2JozWJshmEHcf5H"
api_key = "kT9wlcEQpxrgzKO01QZ6uFkdJ"
api_key_secret = "3TAGj5kvVXCFkXqocWzk7ySOx9ozTCkXGNIJG9hVpL5L8L0nTO"

auth = tweepy.OAuthHandler(api_key,api_key_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

def screping(sk, prog, app):
    import csv
    from tweepy import cursor
    from preprocessing import prepro

    import string
    import pandas as pd

    search_key = sk
    nama_file = "./screping/"+search_key+".csv"
    csvFile = open(nama_file, "a+", encoding="utf-8")
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(["waktu","user","tweet"])
    n = 1000
    x = 0

    for tweet in tweepy.Cursor(api.search_tweets,q=search_key, lang="id").items(n):
        # print(tweet.created_at,tweet.author.screen_name,tweet.text)
        csvWriter.writerow([tweet.created_at, tweet.author.screen_name, tweet.text])
        prog['value']+=100/n
        app.update_idletasks()
        print(n)
        
    csvFile.close()
    prepro(sk, nama_file)
        
    