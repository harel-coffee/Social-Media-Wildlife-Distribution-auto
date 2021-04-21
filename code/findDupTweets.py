import mysql.connector
import Levenshtein as lev
import time

#open database connection
def dbConnection():
    mydb = mysql.connector.connect(
        host="",
        user="",
        passwd="",
        database=""
    )

    mycursor = mydb.cursor()
    mydb.autocommit = True

    return mydb, mycursor

# close database connection
def finish(mydb, mycursor):
    mycursor.close()
    mydb.close()


def simpleDupRemoval():
    allRes = {}
    mydb, mycursor = dbConnection()
    sql = "SELECT tweet_id, cleaned_tweet from simple_classif_wildlife where dup is null order by cleaned_tweet;"
    mycursor.execute(sql)
    res = mycursor.fetchall()
    finish(mydb, mycursor)
    for row in res:
        tid = str(row[0]).replace('b','').replace("'","").strip()
        tweet = row[1].strip()
        allRes[tid] = tweet

    alreadyiterated = []
    tomark = []
    for id in allRes.keys():

        for mid in allRes.keys():
            if id != mid and mid not in alreadyiterated:
                ratio_res = lev.ratio(allRes[id].lower(), allRes[mid].lower())
                if ratio_res >= 0.96:
                    #print(ratio_res)
                    #print(id,":",mid)
                    #print(allRes[id],":",allRes[mid])
                    mydb1, mycursor1 = dbConnection()
                    sql_1 = "UPDATE simple_classif_wildlife SET dup = 'partial' WHERE tweet_id = '"+mid+"';"
                    mycursor1.execute(sql_1)
                    finish(mydb1, mycursor1)
                    #print("-------------------")
        alreadyiterated.append(id)



def precentageDuplicates():
    allTokens = {}
    mydb, mycursor = dbConnection()
    sql = "SELECT tweet_id, cleaned_tweet from simple_classif_wildlife where interAnnotated = 'yes';"
    mycursor.execute(sql)
    result = mycursor.fetchall()


    for row in result:
        tid = str(row[0]).replace("b","").replace("'","")
        cl_tweet = row[1].split()
        if len(cl_tweet) > 1:
            allTokens[tid] = cl_tweet

    alreadyiterated = []
    tomark = []
    for id in allTokens.keys():

        for mid in allTokens.keys():
            if id != mid and mid not in alreadyiterated:

                intersection = list(set(allTokens[id]).intersection(allTokens[mid]))
                newintersection = []
                for el in intersection:
                    newintersection.append(el)


                threshold1 = 0.9 * len(allTokens[id])
                threshold2 = 0.9 * len(allTokens[mid])

                if float(len(newintersection)) >= threshold1 and float(len(newintersection)) <= threshold2:
                    tomark.append(id)
                   

                elif float(len(newintersection)) >= threshold2 and float(len(newintersection)) <= threshold1:
                    tomark.append(mid)

                elif float(len(newintersection)) >= threshold1 and float(len(newintersection)) >= threshold2:
                    tomark.append(mid)
                    
        alreadyiterated.append(id)

    removedup = list(set(tomark))
    print(len(removedup))


def main():
    #findFullDuplicates()
    #findMoreDup()
    simpleDupRemoval()
    precentageDuplicates()



main()