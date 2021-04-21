import mysql.connector
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile, SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold


# open database connection
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


def getStopWords():
    stopwords = []
    mydb, mycursor = dbConnection()
    sql = "select distinct lemma " \
          "from tweets_tokens " \
          "where ne_tag in ('PERCENT','MONEY','DATE','NUMBER','ORDINAL','DURATION') " \
          "and lemma not in ('weekend','summer', 'saturday','april','may','30dayswild', " \
          "'spring','july','sunday','june','friday','@dknott8','bank', " \
          "'holiday','Monday','Wednesday','eyeshadow','tuesday','@nationaltrust', " \
          "'http://tipperarycloudaccesshost/p=19167','lookbook','september', " \
          "'cool','@groundwork9','january', 'midcenturymodern','August','Autumn','StreamerNetwork','twitch','blackops4', " \
          "'october','november','beech','Chilterns','fallout76','HammondsWood','landscape','lowsun', " \
          "'gaming','rainbow6siege','Winter','qualifier','3timesisacharm','#ps4','8ink','Thursday','LastYearTheNightmare', " \
          "'discord','twitchaffiliate','twitchstreamer','teitchtvpictwittercom/lk2zsaqzdt','Monday-Friday', " \
          "'midday','2piece','200mitosfortnite','snipe','#astroa50','@zaitr0s','2fwatch','3fv', " \
          "'26feature','3diqxwotcjg74','amazon','2fclipstwitchtv','2fempathictendersharkrlytho','3ftt_medium', " \
          "'londonnaturerobin','teamzeal','u4ik','2foptimistichappysmoothiepicomause', " \
          "'2fantediluvianhardantelopehotpokket','brash0000','poggers','pubg','#bf5','link', " \
          "'SupportSmallStreamers','faucxbg','Div','100list','SmallStreamersConnect','robinsrspbuk','late', " \
          "'CallOfDuty','blackout','supportsmallstreamer','HappyFathersDay','streampromote1','Christmas', " \
          "'TheyAreBillions','420twitch','madcupid6','twitchkitten','overwatch','csgo','destiny2','r3dfalkon', " \
          "'gardenersworldpictwittercom','HailHydra','december','lfcmay25','Feb','ErithacusRubecula','ukbird', " \
          "'Jan','February','March','Outdoors','#greattit','Jul','Mummy','Aug','Sep','10minsinthegarden','Blackbird', " \
          "'Saturdays','eyesixxgfxpictwittercom/ufdxbptsl1','chaffinch','supper') " \
          "and lemma not like 'http:%' " \
          "and lemma not like 'photography%' " \
          "and lemma not like '%pictwittercom%' " \
          "and lemma not like '%#' " \
          "and lemma not like '%@' " \
          "and lemma not like 'pictwittercom%';"
    mycursor.execute(sql)
    result = mycursor.fetchall()
    finish(mydb, mycursor)

    for row in result:
        token = row[0].strip().lower()

        stopwords.append(token)
        stopwords.append('make')
        stopwords.append('be')
        stopwords.append('get')
        stopwords.append('here')
        # stopwords.append('flickr')
        # stopwords.append('twitter')
        # stopwords.append('pictwitter')
        # stopwords.append('#flickr')
        # stopwords.append('#twitter')
        # stopwords.append('#pictwitter')

    return stopwords


def getVectorsX():
    stopwords = getStopWords()
    x_values = {}
    dub = []
    ids = []
    mydb, mycursor = dbConnection()
    # sql = "SELECT tweet_id,tweet from tweets_embeddings;"
    # sql = "select tweet_id, cleaned_tweet from simple_classif_wildlife where dup is null order by cleaned_tweet;"
    # sql = "select tweet_id, cleaned_tweet from simple_classif_wildlife where dup is null;"
    sql = "select tweet_id, cleaned_tweet from simple_classif_wildlife where interAnnotated = 'yes' and dup is null;"
    #sql = "select tweet_id, tweets_terms from simple_classif_wildlife where interAnnotated = 'yes' and dup is null;"
    # sql = "SELECT tweet_id,cleaned_tweet from tweets_test;"
    mycursor.execute(sql)
    result = mycursor.fetchall()
    finish(mydb, mycursor)
    for row in result:
        tweet_id = str(row[0]).replace('b', '').replace("'", "").strip()
        tweet = str(row[1])
        # tweet = " ".join(l for l in tweet if not l.isdigit())
        tokens = []
        for token in tweet.split():
            if token not in stopwords:
                if 'pictwittercom' in token:
                    token = 'pictwittercom'
                if 'clipstwitchtv' in token:
                    token = 'clipstwitchtv'
                token = token.replace('#', '').replace('@', '')
                tokens.append(token)

        tokens_nodup = list(set(tokens))
        # print(tokens_nodup)
        new_tweet = " ".join(t for t in tokens_nodup)

        x_values[tweet_id] = new_tweet
        # x_values[tweet_id] = tweet
        # with open("XallOrdered.txt","a") as f:
        #    f.write(new_tweet+'\n')

    df_X = pd.DataFrame.from_dict(x_values.items())

    return df_X


def getVectorsX_train():
    x_values = {}
    mydb, mycursor = dbConnection()

    sql = "SELECT id, tweet from trainBERTOriginal;"
    mycursor.execute(sql)
    result = mycursor.fetchall()
    finish(mydb, mycursor)
    for row in result:
        tweet_id = str(row[0]).replace('b', '').replace("'", "").strip()
        tweet = str(row[1])

        x_values[tweet_id] = tweet

    df_X = pd.DataFrame.from_dict(x_values.items())

    return df_X


def getVectorsX_test():
    x_values = {}
    mydb, mycursor = dbConnection()

    sql = "SELECT id, tweet from testSetcomparison;"
    mycursor.execute(sql)
    result = mycursor.fetchall()
    finish(mydb, mycursor)
    for row in result:
        tweet_id = str(row[0]).replace('b', '').replace("'", "").strip()
        tweet = str(row[1])

        x_values[tweet_id] = tweet

    df_X = pd.DataFrame.from_dict(x_values.items())

    return df_X


def getVectorsY():
    y_values = {}
    mydb, mycursor = dbConnection()
    # sql = "select tweet_id,label from simple_classif_wildlife where dup is null;"
    sql = "select tweet_id,label from simple_classif_wildlife where interAnnotated = 'yes' and dup is null;"
    # sql = "select tweet_id,label from tweets_test;"
    mycursor.execute(sql)
    result = mycursor.fetchall()
    finish(mydb, mycursor)
    countY = 0
    countN = 0
    for row in result:
        tweet_id = str(row[0]).replace('b', '').replace("'", "").strip()
        label = str(row[1]).replace('b', '').replace("'", "").strip()
        if label == 'yes':
            label = 1
            countY = countY + 1
        if label == 'no':
            label = 0
            countN = countN + 1

        y_values[tweet_id] = label

    # print(countY)
    # print(countN)
    df_y = pd.DataFrame.from_dict(y_values.items())
    return df_y


def getVectorsY_train():
    y_values = {}
    mydb, mycursor = dbConnection()
    # sql = "select tweet_id,label from simple_classif_wildlife where dup is null;"
    # sql = "select tweet_id,label from simple_classif_wildlife where interAnnotated = 'yes' and dup is null;"
    sql = "SELECT id, label from trainBERTOriginal;"
    mycursor.execute(sql)
    result = mycursor.fetchall()
    finish(mydb, mycursor)
    for row in result:
        tweet_id = str(row[0]).replace('b', '').replace("'", "").strip()
        label = int(row[1])
        y_values[tweet_id] = label

    # print(countY)
    # print(countN)
    df_y = pd.DataFrame.from_dict(y_values.items())
    return df_y


def getVectorsY_test():
    y_values = {}
    mydb, mycursor = dbConnection()
    # sql = "select tweet_id,label from simple_classif_wildlife where dup is null;"
    # sql = "select tweet_id,label from simple_classif_wildlife where interAnnotated = 'yes' and dup is null;"
    sql = "SELECT id, actualLabel from testSetcomparison;"
    mycursor.execute(sql)
    result = mycursor.fetchall()
    finish(mydb, mycursor)
    for row in result:
        tweet_id = str(row[0]).replace('b', '').replace("'", "").strip()
        label = int(row[1])
        y_values[tweet_id] = label

    # print(countY)
    # print(countN)
    df_y = pd.DataFrame.from_dict(y_values.items())
    return df_y


# split tweets into training and testing set
def splitData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    return X_train, X_test, y_train, y_test


# create count feature vectors
def createFeatureVector_count(X_df, X_train, X_test):
    vectorizer = CountVectorizer(max_features=1000,ngram_range=(1,2))
    #vectorizer = CountVectorizer(ngram_range=(1, 2))
    #vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=1000)
    vectorizer.fit(X_df[1])
    feat_names = vectorizer.get_feature_names()
    vector_train = vectorizer.transform(X_train[1])
    vector_test = vectorizer.transform(X_test[1])

    return vector_train, vector_test, feat_names


def TFIDFVectors(X_df, X_train, X_test):
    vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True, max_features=1000, stop_words="english",ngram_range=(1, 2))
    #vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True, stop_words="english",ngram_range=(1, 2))
    X = vectorizer.fit_transform(X_df[1])
    vector_train = vectorizer.transform(X_train[1])
    vector_test = vectorizer.transform(X_test[1])
    feat_names = vectorizer.get_feature_names()

    return vector_train, vector_test, feat_names


def createModel(vector_train, y_train):
    # model = ComplementNB(alpha=1.0, fit_prior=False, class_prior=None)
    # model = GaussianNB()
    model = LogisticRegression(random_state=2)
    # model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
    # model = LogisticRegression(random_state=0, solver='lbfgs',class_weight = 'balanced',max_iter = 1000)
    # model = svm.LinearSVC()
    # model = svm.SVC(gamma='scale',kernel='linear')

    model.fit(vector_train.toarray(), y_train[1])
    filename = 'lib/please.sav'
    pickle.dump(model, open(filename, 'wb'))


def predictModel(y_test, vector_test):
    model = pickle.load(open('lib/please.sav', 'rb'))
    y_predict = model.predict(vector_test)
    zipped = zip(y_test[0], y_test[1], y_predict)
    return y_predict, zipped


def evaluate(y_test, y_predict, X_test):
    # print("precision: ",metrics.precision_score(y_test[1], y_predict))
    p = metrics.precision_score(y_test[1], y_predict)
    # print("recall: ",metrics.recall_score(y_test[1], y_predict))
    r = metrics.recall_score(y_test[1], y_predict)
    # print("f_1 measure: ",metrics.f1_score(y_test[1], y_predict))
    f1 = metrics.f1_score(y_test[1], y_predict)
    # print("accuracy: ", metrics.accuracy_score(y_test[1], y_predict))
    a = metrics.accuracy_score(y_test[1], y_predict)
    return p, r, f1, a

    # classif_rep = classification_report(y_test[1].to_numpy(), y_predict, target_names=['no','yes'])
    # print(classif_rep)


def show_most_informative_features(feat_names):
    clf = pickle.load(open('lib/please.sav', 'rb'))
    class_labels = clf.classes_
    top10 = np.argsort(clf.coef_[0])[-20:]
    print("%s: %s" % (class_labels, ",".join(feat_names[j] for j in top10)))


def wm2df(X_train, vector_train, feat_names):
    df = pd.DataFrame(data=vector_train.toarray(), index=X_train[0], columns=feat_names)
    return (df)


def selectFeatures(df_train, df_test, y_train):
    # selector = SelectPercentile(chi2, percentile=30)
    selector = SelectKBest(chi2, k=1000)
    selector.fit(df_train, y_train)
    features_train_transformed = selector.transform(df_train)
    features_test_transformed = selector.transform(df_test)
    # lsvc = ComplementNB(alpha=1.0, fit_prior=False, class_prior=None).fit(df_train, y_train)
    # model = SelectFromModel(lsvc, prefit=True)
    # model.fit(df_train, df_test)
    # features_train_transformed = model.transform(df_train)
    # features_test_transformed = model.transform(df_test)

    return features_train_transformed, features_test_transformed


def main():
    X_df = getVectorsX()
    y_df = getVectorsY()

    # X_train = getVectorsX_train()
    # y_train = getVectorsY_train()

    # X_test = getVectorsX_test()
    # y_test = getVectorsY_test()

    # X_df = pd.concat([X_train, X_test])
    # y_df = pd.concat([y_train, y_test])

    # X_train, X_test, y_train, y_test = splitData(X_df, y_df)
    # X_train.to_csv('X_train.csv')
    # y_train.to_csv('y_train.csv')
    # X_test.to_csv('X_test.csv')
    # y_test.to_csv('y_test.csv')

    # zipped = list(zip(X_train[0],X_train[1], y_train[1]))
    # print(zipped)
    # for i in range(0,len(zipped)):
    #    print(zipped[i][0])
    #    print(zipped[i][1])
    #    print(zipped[i][2])
    #    print("----------------------")

    # vector_train, vector_test, feat_names = createFeatureVector_count(X_df, X_train, X_test)
    # vector_train, vector_test, feat_names = TFIDFVectors(X_df, X_train, X_test)

    # createModel(vector_train, y_train)
    # y_predict, zipped_1 = predictModel(y_test, vector_test.toarray())

    # mydb, mycursor = dbConnection()
    # for i in zipped_1:
    #    sql = "UPDATE testSetcomparison SET baseline = '"+str(i[2]).strip()+"' WHERE id = '"+str(i[0])+"';"
    #    mycursor.execute(sql)
    # finish(mydb, mycursor)

    # p,r,f1,a = evaluate(y_test, y_predict, X_test)
    # print(p,r,f1,a)
    # show_most_informative_features(feat_names)

    # print("test versus predict")
    # for i in zipped_1:
    #    print(i)

    # show_most_informative_features(feat_names)

    # WITH FEATURE SELECTION
    # vector_train, vector_test, feat_names = createFeatureVector_count(X_df, X_train, X_test)
    # vector_train, vector_test, feat_names = TFIDFVectors(X_df, X_train, X_test)
    # df_train = wm2df(X_train, vector_train, feat_names)
    # df_test = wm2df(X_test, vector_test, feat_names)
    # features_train_transformed,features_test_transformed = selectFeatures(df_train, df_test, list(y_train[1]))

    # createModel(features_train_transformed, y_train)
    # y_predict, zipped_1 = predictModel(y_train, features_test_transformed)
    # evaluate(y_test, y_predict, X_test)
    # show_most_informative_features(feat_names)

    cvscores = []
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    cvscores_p = []
    cvscores_r = []
    cvscores_f1 = []
    cvscores_a = []
    for train, test in kfold.split(X_df[1], y_df[1]):
        x_train, x_test = X_df.iloc[train], X_df.iloc[test]
        y_train, y_test = y_df.iloc[train], y_df.iloc[test]
        #vector_train, vector_test, feat_names = createFeatureVector_count(X_df, x_train, x_test)
        vector_train, vector_test, feat_names = TFIDFVectors(X_df, x_train, x_test)
        # scores = createModel(x_train[1], encoded_Y_train, x_test[1], encoded_Y_test)
        createModel(vector_train, y_train)
        y_predict, zipped_1 = predictModel(y_train, vector_test.toarray())
        p, r, f1, a = evaluate(y_test, y_predict, x_test)
        cvscores_p.append(p * 100)
        cvscores_r.append(r * 100)
        cvscores_f1.append(f1 * 100)
        cvscores_a.append(a * 100)
        # cvscores.append(scores[1] * 100)

    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_p), np.std(cvscores_p)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_r), np.std(cvscores_r)))
    print("F1: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_f1), np.std(cvscores_f1)))
    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_a), np.std(cvscores_a)))

main()