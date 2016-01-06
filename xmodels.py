
from multiprocessing import Pool 
import numpy as np

__all__ = ["Recommender", "Evaluation"]

class Recommender(object):
    def __init__(self, matrix):
        super(Recommender, self).__init__()
        self.matrix = matrix
        self.num_users = matrix.shape[0]
        self.num_items = matrix.shape[1]

    def train(self, before=None, after=None):
        raise NotImplementedError

    def predict(self, user, item):
        raise NotImplementedError

    def recommend(self, user, num=5, ruleout=True):
        scores = []
        for poi in xrange(self.num_items):
            scores.append((poi, self.predict(user, poi)))
        scores.sort(key=lambda x: x[1], reverse=True)

        if self.matrix is not None and ruleout:
            ruleouts = set(np.nonzero(self.matrix[user])[1])
        else:
            ruleouts = set()

        result = []
        for poi, score in scores:
            if poi in ruleouts:
                continue
            result.append(poi)
            if len(result) >= num:
                break
        return result 

        
def num_hits(args):
    evaluation, user = args
    return len(evaluation.hits(user))


class Evaluation(object):
    def __init__(self, matrix, filter_matrix=None, N=5, model=None, _pool_num=6):
        self.matrix = matrix
        self.N = N
        self.model = model
        self.filter_matrix = filter_matrix
        self._pool_num = _pool_num
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

    def hits(self, user, model=None):
        if model is not None:
            self.model = model
        pois = set(np.nonzero(self.matrix[user])[1])
        if len(pois) <= 0:
            return []

        scores = []
        for poi in xrange(self.num_items):
            scores.append((poi, self.model.predict(user, poi)))
        scores.sort(key=lambda x: x[1], reverse=True)
        if self.filter_matrix is None:
            return [poi for poi, score in scores[: self.N] if poi in pois]
        else:
            result = []
            ruleouts = set(np.nonzero(self.filter_matrix[user])[1])
            n = 0
            for poi, score in scores:
                if poi in ruleouts:
                    continue
                if poi in pois: 
                    result.append(poi)
                n += 1
                if n >= self.N:
                    break
            return result 

    def test(self, model=None):
        if model is not None:
            self.model = model
        # use thread pool
        def prepare():
            for user in xrange(self.num_users):
                yield (self, user)

        pool = Pool(self._pool_num)
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

