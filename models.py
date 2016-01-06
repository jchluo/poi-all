
import time
import numpy as np


__all__ = ["Recommender", "Evaluation"]

class Recommender(object):
    def __init__(self):
        pass

    def train(self, before=None, after=None):
        pass

    def predict(self, user, item):
        pass


class Evaluation(object):
    def __init__(self, matrix, N=5):
        self.matrix = matrix
        self.N = N
        self.num_users = matrix.shape[0]
        self.num_items = matrix.shape[1]
        # valid users
        self.valids = 0 
        for i in xrange(self.num_users):
            pois = np.nonzero(self.matrix[i])[1]
            if len(pois) > 0:
                self.valids += 1
        if self.valids == 0:
            raise ValueError("Test checkin set should not be empty.")
        #self.valids = self.num_users

    def hits(self, model, user):
        pois = set(np.nonzero(self.matrix[user])[1])
        if len(pois) <= 0:
            return []

        scores = []
        for poi in xrange(self.num_items):
            scores.append((poi, model.predict(user, poi)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [poi for poi, score in scores[: self.N] if poi in pois]

    def test(self, model):
        nhits = 0
        _recall = 0.0
        t0 = time.time()
        for i in xrange(self.num_users):
            pois = np.nonzero(self.matrix[i])[1]
            if len(pois) <= 0:
                continue
            t1 = time.time()
            print "%i time: %f" % (i, t1 - t0)

            matchs = self.hits(model, i)
            nhits += len(matchs)
            _recall += float(len(matchs)) / len(pois)

        prec = float(nhits) / (self.valids * self.N)
        _recall = float(_recall) / self.valids 
        print "recall   : %.4f" % _recall
        print "precision: %.4f" % prec
        return (_recall, prec)

    def recall(self, model):
        _recall = 0.0 
        for i in xrange(self.num_users):
            pois = np.nonzero(self.matrix[i])[1]
            if len(pois) <= 0:
                continue

            nhits = self.hits(model, i)
            _recall += float(len(nhits)) / len(pois)

        _recall = float(_recall) / self.valids 
        print "recall   : %.4f" % _recall
        return _recall
        
    def precision(self, model):
        nhits = 0
        for i in xrange(self.num_users):
            pois = np.nonzero(self.matrix[i])[1]
            if len(pois) <= 0:
                continue

            nhits += len(self.hits(model, i))
        prec = float(nhits) / (self.valids * self.N)
        print "precision: %.4f" % prec
        return prec

