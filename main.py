
import matplotlib.pyplot as plt
import numpy as np
from wmf import ImplicitMF 
from utils import *
from models import *

setup_log()
fn = Filename("foursquare")
mf = read_model("output/model_29.pkl")
test_matrix = load_matrix(fn.test)

test = Evaluation(test_matrix, model=mf, users=range(10))
test.test()
exit(0)

x = []
y = []
s = []
for i in xrange(100):
    recs = mf.recommend(i)
    m = 5
    for rec in recs:
        x.append(i)
        y.append(rec)
        s.append(m*10)
        m -= 1

x = np.array(x)
y = np.array(y)
plt.scatter(x, y, s=s)
plt.savefig("f.png")

