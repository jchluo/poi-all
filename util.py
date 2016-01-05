
import time
import cPickle
import numpy as np
import scipy.sparse as sparse

def get_data_name(dataset, parent="."):
    return "%s/datasets/%s/data.txt" % (parent, dataset)


def get_train_name(dataset, parent="."):
    return "%s/datasets/%s/train.txt" % (parent, dataset)


def get_test_name(dataset, parent="."):
    return "%s/datasets/%s/test.txt" % (parent, dataset)


def load_matrix(filename):
    t0 = time.time()

    row = []
    col = []
    data = []

    checkins = 0
    users = set()
    items = set()

    for i, line in enumerate(open(filename, 'r')):
        params = line.strip().split('\t')
        user = int(params[0][5:])
        item = int(params[1][4:])
        frequence = 1

        row.append(user)
        col.append(item)
        data.append(frequence)
        
        checkins += 1
        users.add(user)
        items.add(item)

    matrix = sparse.csr_matrix((data, (row, col)), shape=(max(users) + 1, max(items) + 1))
    t1 = time.time()
    print "load file %s" % filename
    print "load %i checkins" % checkins 
    print "load %i users" % len(users)
    print "load %i items" % len(items)
    print 'time %.4f seconds' % (t1 - t0)
    print 'Finished loading.' 
    return matrix


def save_model(model, filename):
    f = open(filename, "w") 
    cPickle.dump(model, f)
    f.close()


def read_model(filename):
    return cPickle.load(filename)


