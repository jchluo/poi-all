
from wmf import ImplicitMF 
from utils import *
if __name__ == "__main__":
    model = read_model("output/model.pkl")
    print model.predict(0, 0)
    
