import pickle
import numpy as np

with open('save/target_params_py3.pkl', 'rb') as f:
    data = pickle.load(f)
    npdata = np.array(data)

    print(npdata.shape)
    print(npdata)