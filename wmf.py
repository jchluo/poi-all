
import time
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

from xmodels import *
from util import *

class ImplicitMF(Recommender):
    def __init__(self, matrix, num_factors=10, num_iterations=30,
                 reg_param=0.8):
        self.matrix = matrix
        self.num_users = matrix.shape[0]
        self.num_items = matrix.shape[1]
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param
        self.iteration = 0 
        # init factor
        self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))

    def train(self, before=None, after=None):
        while self.iteration < self.num_iterations:
            self.iteration += 1 
            t0 = time.time()
            # call back before hook
            if before is not None:
                before(self)
            #print 'Solving for user vectors...'
            self.user_vectors = self.iteration(True, sparse.csr_matrix(self.item_vectors))
            #print 'Solving for item vectors...'
            self.item_vectors = self.iteration(False, sparse.csr_matrix(self.user_vectors))

            # call back the after hook
            if after is not None:
                after(self)

            t1 = time.time()
            print 'iteration %i finished in %f seconds' % (self.iteration, t1 - t0)

    def predict(self, user, item):
        return self.user_vectors[user].T.dot(self.item_vectors[item])

    def iteration(self, user, fixed_vecs):
        num_solve = self.num_users if user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        YTY = fixed_vecs.T.dot(fixed_vecs)
        eye = sparse.eye(num_fixed)
        lambda_eye = self.reg_param * sparse.eye(self.num_factors)
        solve_vecs = np.zeros((num_solve, self.num_factors))

        for i in xrange(num_solve):
            if user:
                matrix_i = self.matrix[i].toarray()
            else:
                matrix_i = self.matrix[:, i].T.toarray()
            CuI = sparse.diags(matrix_i, [0])
            pu = matrix_i.copy()
            pu[np.where(pu != 0)] = 1.0
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            solve_vecs[i] = xu

        return solve_vecs

        
        
    
if __name__ == "__main__":
    mtrain = load_matrix(get_train_name("foursquare"))
    mtest = load_matrix(get_test_name("foursquare"))
    mf = ImplicitMF(mtrain)
    eva = Evaluation(mtest)
    mf.train(before=eva.test)
