
from wmf import ImplicitMF 
from utils import *
mf = read_model("output/model_2.pkl")
print mf.predict(0, 0)
print mf.recommend(0)
