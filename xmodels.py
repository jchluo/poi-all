
from multiprocessing import Pool 
import numpy as np

__all__ = ["Recommender", "Evaluation"]

class Recommender(object):
    def __init__(self):
        pass

    def train(self, before=None, after=None):
        pass

    def predict(self, user, item):
        pass


def num_hits(args):
    evaluation, model, user = args
    return len(evaluation.hits(model, user))


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
        pois = np.nonzero(self.matrix[user])[1]
        if len(pois) <= 0:
            return []

        scores = []
        for poi in xrange(self.num_items):
            scores.append((poi, model.predict(user, poi)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [poi for poi, score in scores[: self.N] if poi in pois]

    def test(self, model):
        # use thread pool
        def prepare():
            for user in xrange(self.num_users):
                yield (self, model, user)

        pool = Pool(4)
        matchs = pool.map(num_hits, prepare()) 
        pool.close()
        pool.join()
        
        nhits = sum(matchs)
        _recall = 0.0
        for i in xrange(self.num_users):
            if matchs[i] > 0:
                pois = np.nonzero(self.matrix[i])[1]
                if len(pois) > 0:
                    _recall += float(matchs[i]) / len(pois)

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

