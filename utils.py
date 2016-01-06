
import time
import cPickle
import logging
import numpy as np
import scipy.sparse as sparse

log = logging.getLogger(__name__)

class Filename(object):
    def __init__(self, dataset, parent="."):
        self._dataset = dataset
        self.parent = parent
        self.dataset = "%s/datasets/%s/data.txt" % (parent, dataset)
        self.train = "%s/datasets/%s/train.txt" % (parent, dataset)
        self.test = "%s/datasets/%s/test.txt" % (parent, dataset)
        self.locations = "%s/datasets/%s/locations.txt" % (parent, dataset)

    def log(self, model_name):
        return "%s/log/%s-%s.log" % (self.parent, self._dataset, model_name)

    
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
    log.debug("load file %s" % filename)
    log.debug("load %i checkins, %i users, %i pois." % (count, len(users), len(items)))
    log.debug('time %.4f seconds' % (t1 - t0))
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
 

def setup_log(filename="debug.log"):
    #sformat = "%(asctime)s %(filename)s[line:%(lineno)d]"\
    #            " %(levelname)s %(message)s"
    sformat = "%(asctime)s %(filename)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.DEBUG,
                        format=sformat,
                        filename=filename,
                        filemode="a")

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(sformat))
    logging.getLogger('').addHandler(console)
    log.info("new session")

    
if __name__ == "__main__":
    locs = poi_locations(Filename("foursquare").train)
    f = open(Filename("foursquare").locations, "w")
    for poi in locs:
        print >> f, "%d\t%.16f\t%.18f" % (poi, locs[poi][0], locs[poi][1])
    f.close()

#print poi_locations(Filename("foursquare").train)
