
from utils import *

def count_max(filename):
    max_user = -1
    max_item = -1
    with open(filename) as in_file:
        for line in in_file:
            data = line.strip().split()
            try:
                user = int(data[0])
                item = int(data[1])
            except:
                user = int(data[0][5: ])
                item = int(data[1][4: ])
            if user > max_user:
                max_user = user
            if item > max_item:
                max_item = item
    print "num user: %d" % (max_user + 1)
    print "num item: %d" % (max_item + 1)


def split(dataset, train=0.7):
    filename = "./datasets/%s/data.txt" % dataset
    trainname = "./datasets/%s/train.txt" % dataset
    testname = "./datasets/%s/test.txt" % dataset

    users = {}
    with open(filename) as in_file:
        for line in in_file:
            data = line.strip().split()
            user = int(data[0])
            item = int(data[-1])
            if user not in users:
                users[user] = {}
            users[user][item] = users[user].get(item, 0) + 1

    # ckeck
    checkeds = {}
    for user in users:
        if len(users[user]) >= 20:
            checkeds[user] = users[user]

    # assing id to user and item
    user_id = {}
    item_id = {}
    uid = 0
    iid = 0
    for user in checkeds:
        user_id[user] = uid
        uid += 1
        for item in checkeds[user]:
            if item not in item_id:
                item_id[item] = iid
                iid += 1
        
    trainfile = open(trainname, 'w')
    testfile = open(testname, 'w')
    for user in checkeds:
        item_counts = checkeds[user].items()
        n = int(round(train * len(item_counts)))
        for item, count in item_counts[: n]:
            print >> trainfile, "%d\t%d\t%d" % (user_id[user], item_id[item], count)
        for item, count in item_counts[n:]:
            print >> testfile, "%d\t%d\t%d" % (user_id[user], item_id[item], count)
    trainfile.close()
    testfile.close()
            

if __name__ == "__main__":
    #split("gowalla")
    count_max(get_train_name("foursquare"))

