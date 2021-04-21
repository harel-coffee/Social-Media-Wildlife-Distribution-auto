from fse.models import uSIF
#from transformers import *
import numpy as np
#from fse.models import SIF
from fse import IndexedList
from gensim.models import KeyedVectors
import io
from sklearn.feature_extraction.text import TfidfVectorizer

#import torch
#from transformers import *

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
import string
import pandas as pd
import mysql.connector


def dbConnection():
    mydb = mysql.connector.connect(
        host="csmysql.cs.cf.ac.uk",
        user="c1114882",
        passwd="thom9055",
        database="c1114882"
    )

    mycursor = mydb.cursor()
    mydb.autocommit = True

    return mydb, mycursor

# close database connection
def finish(mydb, mycursor):
    mycursor.close()
    mydb.close()


def calcUSIF():

    tweets = []
    mydb, mycursor = dbConnection()
    #sql = "SELECT tweet FROM tweets_embeddings_nodup where fastText_usif is null;"
    #sql = "SELECT tweet FROM tweets_embeddings_nodup where w2v_usif is null;"
    #sql = "SELECT tweet FROM tweets_embeddings_nodup where corpus_usif is null;"
    sql = "select emb.tweet " \
          "from tweets_embeddings_nodup emb, simple_classif_wildlife sc " \
          "where sc.dup is null and sc.tweet_id = emb.tweet_id and corpus_usif is null;"
    mycursor.execute(sql)
    result = mycursor.fetchall()
    #finish(mydb, mycursor)
    for row in result:
        tweet = str(row[0]).split(" ")
        tweets.append(tweet)

    #ft = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    #ft = KeyedVectors.load_word2vec_format('crawl-300d-2M-subword.vec')
    #ft = KeyedVectors.load_word2vec_format('model_tweets.vec')
    ft = KeyedVectors.load_word2vec_format('tweets_model.vec')
    model = uSIF(ft, components=10, lang_freq="en")
    model.train(IndexedList(tweets))
    for j in range(0, len(tweets)):
        tokens = []
        for t in tweets[j]:
            tokens.append(t.strip())

        sent_whole = " ".join(t for t in tokens)

        #mydb, mycursor = dbConnection()
        #sql_up = "UPDATE tweets_embeddings SET fastext_sif = '" + str(model.sv[j]) + "' WHERE tweet = '"+sent_whole+"';"
        #sql_up = "UPDATE tweets_embeddings_nodup SET fastText_usif = '" + str(model.sv[j]) + "' WHERE tweet = '" + sent_whole + "';"
        #sql_up = "UPDATE tweets_embeddings_nodup SET w2v_usif = '" + str(model.sv[j]) + "' WHERE tweet = '" + sent_whole + "';"
        #sql_up = "UPDATE tweets_embeddings SET w2v_sif = '" + str(model.sv[j]) + "' WHERE tweet = '" + sent_whole + "';"
        sql_up = "UPDATE tweets_embeddings_nodup SET corpus_usif = '" + str(model.sv[j]) + "' WHERE tweet = '" + sent_whole + "';"
        mycursor.execute(sql_up)
    finish(mydb, mycursor)

def tokenVectorsFastext(fname):
    tokens_ids = {}
    tokens_vec = {}
    mydb, mycursor = dbConnection()
    #sql = "SELECT tweet_id, tweet from tweets_embeddings where tweet_id not in ('1.10817e18','1.10954e18') and fastext_avg is null;"
    #sql = "SELECT tweet_id, tweet from tweets_embeddings_nodup where fastText_avg is null;"
    #sql = "SELECT tweet_id, tweet from tweets_embeddings_nodup where fastText_avg is null;"
    sql = "select emb.tweet_id, emb.tweet " \
          "from tweets_embeddings_nodup emb, simple_classif_wildlife sc " \
          "where sc.dup is null and sc.tweet_id = emb.tweet_id and corpus_avg is null;"

    #sql = "SELECT tweet_id, tweet from tweets_embeddings_nodup where corpus_avg is null;"
    mycursor.execute(sql)
    res = mycursor.fetchall()
    finish(mydb, mycursor)
    for row in res:
        tweet_id = str(row[0]).replace('b', '').replace("'", "")
        list_tokens = []
        tweet = row[1].split()
        for t in tweet:
            t = t.strip().lower()
            list_tokens.append(t)

        #print(tweet_id)
        #print(list_tokens)
        tokens_ids[tweet_id] = list_tokens

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    #n, d = map(int, fin.readline().split())
    count = 0
    for line in fin:
        count = count + 1
        tokens = line.rstrip().split(' ')
        word = tokens[0].lower().replace("-", "").replace("_", "").replace(" ", "").strip()
        vect = str(tokens[1:]).replace("'", "")
        for tid in tokens_ids.keys():
            if word in tokens_ids[tid]:
                #print(word)
                vec_floats = []
                vect = str(vect).replace("'", "")
                vect_list = vect.replace("[", "").replace("]", "").split(",")
                for val in vect_list:
                    val = val.strip()
                    val = float(val)
                    vec_floats.append(val)

                try:
                    tokens_vec[tid].append(vec_floats)
                except KeyError:
                    tokens_vec[tid] = [vec_floats]

    return tokens_vec

def tokenVectorsW2V():
    tokens_ids = {}
    tokens_vec = {}
    mydb, mycursor = dbConnection()
    sql = "SELECT tweet_id, tweet from tweets_embeddings_nodup where w2v_avg is null;"
    mycursor.execute(sql)
    res = mycursor.fetchall()
    finish(mydb, mycursor)
    for row in res:
        tweet_id = str(row[0]).replace('b','').replace("'","")
        list_tokens = []
        tweet = row[1].split()
        for t in tweet:
            t = t.strip().lower()
            list_tokens.append(t)
        tokens_ids[tweet_id] = list_tokens


    wv_from_bin = KeyedVectors.load_word2vec_format('googlenews.vec')
    for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
        word = word.replace(" ","").replace(".","").replace("?","").replace("!","").replace("(","").replace(")","").replace(";","").replace("_","").replace("-","").strip().lower()

        for tid in tokens_ids.keys():
            if word in tokens_ids[tid]:

                vec_floats = []
                vect = str(vector).replace("'", "")
                vect_list = vect.replace("[","").replace("]","").split()
                for val in vect_list:
                    val = val.strip()
                    val = float(val)
                    vec_floats.append(val)

                try:
                    tokens_vec[tid].append(vec_floats)
                except KeyError:
                    tokens_vec[tid] = [vec_floats]

    return tokens_vec



def manualAvgSentEmb():
    #tokens_vec = tokenVectorsFastext('crawl-300d-2M-subword.vec')
    tokens_vec = tokenVectorsFastext('tweets_model.vec')
    #tokens_vec = tokenVectorsW2V()
    print(len(tokens_vec))

    for sent_id in tokens_vec.keys():
        sent2vec = np.mean(tokens_vec[sent_id], axis=0)
        mydb, mycursor = dbConnection()

        #sql = "UPDATE tweets_embeddings_nodup SET fastText_avg = '" + str(sent2vec) + "' WHERE tweet_id = '" + sent_id.strip() + "';"
        #sql = "UPDATE tweets_embeddings_nodup SET w2v_avg = '" + str(sent2vec) + "' WHERE tweet_id = '" + sent_id.strip() + "';"
        sql = "UPDATE tweets_embeddings_nodup SET corpus_avg = '" + str(sent2vec) + "' WHERE tweet_id = '" + sent_id.strip() + "';"
        mycursor.execute(sql)
        mydb.commit()
        finish(mydb, mycursor)

def calcTfIdfScores():
    all_tweets = []
    tweetstokens = {}
    allscores_pertweet = {}
    mydb, mycursor = dbConnection()
    sql = "SELECT tweet_id, tweet FROM tweets_embeddings_nodup;"

    mycursor.execute(sql)
    res = mycursor.fetchall()
    finish(mydb, mycursor)
    for row in res:
        tid = str(row[0]).replace('b', '').replace("'", "")
        tweet_l = row[1].split()
        list_tokens = []
        for t in tweet_l:
            t = t.strip().lower()
            list_tokens.append(t)

        tweet_cl = " ".join(t for t in list_tokens)
        all_tweets.append([tid, tweet_cl])
        tweetstokens[tid] = list_tokens

    tids = []
    corpus = []
    for item in all_tweets:
        tids.append(item[0].strip())
        corpus.append(item[1])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    wordslist = vectorizer.get_feature_names()
    tfidfscores = X.toarray()

    for i in range(0, len(tids)):
        allscores_pertweet[tids[i]] = {}
        for j in range(0, len(tfidfscores[i])):
            score = float(tfidfscores[i][j])
            if wordslist[j] in tweetstokens[tids[i]]:
                allscores_pertweet[tids[i]][wordslist[j]] = score

    return allscores_pertweet,tweetstokens

def buildTfIDfPerSentFastext():
    allscores_pertweet,tweetstokens = calcTfIdfScores()
    aftertfidf = {}
    all_words_corpus = []
    for k in allscores_pertweet.keys():
        for w in allscores_pertweet[k].keys():
            all_words_corpus.append(w)
            try:
                aftertfidf[k].append(w)
            except KeyError:
                aftertfidf[k] = [w]

    set_allwords = list(set(all_words_corpus))

    tokens_vec = {}
    #fin = io.open('crawl-300d-2M-subword.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
    fin = io.open('tweets_model.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
    count = 0
    dictionary_fastext = {}
    fastextwords = []
    for line in fin:
        count = count + 1
        tokens = line.rstrip().split(' ')
        word = tokens[0].lower().replace("-", "").replace("_", "").replace(" ", "").replace("#","").replace("@","").strip()
        if word in set_allwords:

            fastextwords.append(word)
            vect = str(tokens[1:]).replace("'", "").replace("[", "").replace("]", "").split(", ")

            dictionary_fastext[word] = vect

    for tid in aftertfidf.keys():
        for tweet_token in aftertfidf[tid]:
            if tweet_token in fastextwords:
                tfidf_score = float(allscores_pertweet[tid][tweet_token])
                vec_floats = [float(i) * tfidf_score for i in dictionary_fastext[tweet_token]]
                try:
                    tokens_vec[tid].append(vec_floats)
                except KeyError:
                    tokens_vec[tid] = [vec_floats]

    for t_id in tokens_vec.keys():
        #print(t_id)
        #print(tokens_vec[t_id])
        sent2vec = np.mean(tokens_vec[t_id], axis=0)
        mydb, mycursor = dbConnection()
        #sql = "UPDATE tweets_embeddings_nodup SET fastText_tfidf = '" + str(sent2vec) + "' WHERE tweet_id = '"+t_id.strip()+"';"
        sql = "UPDATE tweets_embeddings_nodup SET corpus_tfidf = '" + str(sent2vec) + "' WHERE tweet_id = '" + t_id.strip() + "';"
        mycursor.execute(sql)
        mydb.commit()
        finish(mydb, mycursor)

def buildTfIDfPerSentWord2Vec():
    allscores_pertweet,tweetstokens = calcTfIdfScores()
    aftertfidf = {}
    all_words_corpus = []
    for k in allscores_pertweet.keys():
        for w in allscores_pertweet[k].keys():
            all_words_corpus.append(w)
            try:
                aftertfidf[k].append(w)
            except KeyError:
                aftertfidf[k] = [w]

    set_allwords = list(set(all_words_corpus))

    tokens_vec = {}

    count = 0
    dictionary_fastext = {}
    fastextwords = []
    wv_from_bin = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
        word = word.replace(" ", "").replace(".", "").replace("?", "").replace("!", "").replace("(", "").replace(")","").replace(";", "").replace("_", "").replace("-", "").replace("#","").replace("@","").strip().lower()
        if word in set_allwords:
            #print("word: ",word)

            fastextwords.append(word)
            vect = str(vector).replace("'", "").replace("[", "").replace("]", "").split()
            #print("vect: ", vect)
            dictionary_fastext[word] = vect

    for tid in aftertfidf.keys():
        for tweet_token in aftertfidf[tid]:
            if tweet_token in fastextwords:
                tfidf_score = float(allscores_pertweet[tid][tweet_token])
                vec_floats = [float(i.strip()) * tfidf_score for i in dictionary_fastext[tweet_token]]
                try:
                    tokens_vec[tid].append(vec_floats)
                except KeyError:
                    tokens_vec[tid] = [vec_floats]

    for t_id in tokens_vec.keys():
        #print(t_id)
        #print(tokens_vec[t_id])
        sent2vec = np.mean(tokens_vec[t_id], axis=0)
        mydb, mycursor = dbConnection()
        sql = "UPDATE tweets_embeddings_nodup SET w2v_tfidf = '" + str(sent2vec) + "' WHERE tweet_id = '" + t_id.strip() + "';"
        mycursor.execute(sql)
        mydb.commit()
        finish(mydb, mycursor)

def createBert():
    sentences = {}
    #sql = "SELECT sent_id, sent FROM sentvectors_embeddings_bert where simple_vec is null;"
    mydb, mycursor = dbConnection()
    sql = "SELECT tweet_id,tweet from tweets_embeddings_nodup where bert is null;"
    mycursor.execute(sql)
    res = mycursor.fetchall()
    finish(mydb, mycursor)

    for row in res:
        sid = str(row[0]).replace('b', '').replace("'", "")
        #print(sid)
        sent = str(row[1]).lower()
        #print(sent)
        bert_final_vector = getSentBertVect(sent)
        norm_vector = []
        for val in bert_final_vector:
            val = float(val)
            normalised_val = round(float(val), 4)
            #print(normalised_val)
            norm_vector.append(normalised_val)

        mydb, mycursor = dbConnection()
        sql_up = "UPDATE tweets_embeddings_nodup SET bert = '"+str(norm_vector)+"' WHERE tweet_id = '"+sid+"';"
        mycursor.execute(sql_up)
        mydb.commit()
        finish(mydb, mycursor)

def getSentBertVect(input_sentence):
    model_class, tokenizer_class, pretrained_weights = BertModel, BertTokenizer, 'bert-base-uncased'
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    input_ids = torch.tensor([tokenizer.encode(input_sentence, add_special_tokens=True)])

    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]

    bert_final_vector= (last_hidden_states[0][-1])
    return bert_final_vector

def main():
    calcUSIF()
    manualAvgSentEmb()
    buildTfIDfPerSentFastext()
    buildTfIDfPerSentWord2Vec()
    createBert()

main()