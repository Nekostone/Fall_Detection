import pickle
import numpy as np
rfc = pickle.load(open("/home/chongyicheng/Capstone/Fall_Detection/randomforest.sav", 'rb'))

testnp = np.load("/home/chongyicheng/Capstone/Fall_Detection/ml/temp/105.npy",allow_pickle=True)
print("Data is actually {0}".format(testnp[1]))
prediction = rfc.predict(testnp[0])
print("Prediction is actually {0}".format(prediction))
