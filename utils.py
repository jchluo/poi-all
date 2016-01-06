
import time
import cPickle
import numpy as np
import scipy.sparse as sparse

class Filename(object):
    def __init__(self, dataset, parent="."):
        self.dataset = "%s/datasets/%s/data.txt" % (parent, dataset)
        self.train = "%s/datasets/%s/train.txt" % (parent, dataset)
        self.test = "%s/datasets/%s/test.txt" % (parent, dataset)
        self.locations = "%s/datasets/%s/locations.txt" % (parent, dataset)

    
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

    count = 0
    users = set()
    items = set()
    checkins  = set()
    
    for i, line in enumerate(open(filename, 'r')):
        params = line.strip().split('\t')
        user = int(params[0])
        item = int(params[1])
        frequence = 1

        if (user, item) in checkins:
            continue
        checkins.add((user, item))

        row.append(user)
        col.append(item)
        data.append(frequence)
        
        count += 1
        users.add(user)
        items.add(item)

    matrix = sparse.csr_matrix((data, (row, col)), shape=(max(users) + 1, max(items) + 1))
    t1 = time.time()
    print "load file %s" % filename
    print "load %i checkins" % count 
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
    f = open(filename, "r")
    model = cPickle.load(f)
    f.close()
    return model


def poi_locations(filename):
    locations = {}
    with open(filename) as in_file:
        for line in in_file:
            params = line.strip().split('\t')
            item = int(params[1])
            lat, lon = params[2].split(",")
            lat = float(lat)
            lon = float(lon)
            locations[item] = (lat, lon)
    return locations


def load_locations(filename):
    locations = {}
    with open(filename) as in_file:
        for line in in_file:
            params = line.strip().split('\t')
            item = int(params[0])
            lat = float(params[1])
            lon = float(params[2])
            locations[item] = (lat, lon)
    return locations
 

if __name__ == "__main__":
    locs = poi_locations(Filename("foursquare").train)
    f = open(Filename("foursquare").locations, "w")
    for poi in locs:
        print >> f, "%d\t%.16f\t%.18f" % (poi, locs[poi][0], locs[poi][1])
    f.close()

#print poi_locations(Filename("foursquare").train)
