import tensorflow as tf
from tensorflow import keras

tf.compat.v1.reset_default_graph()

import numpy as np
import pickle

#load symbol data
#test_symbol_data=np.loadtxt("SR-ARE-score\\names_labels.txt",dtype='str',delimiter=',')
#new_symbol = []

#parse and change to float
#for x in test_symbol_data:
    #y=np.delete(x, 0)
    #new_symbol.append(int(y))

#change back to numpy
#test_symbol=np.array(new_symbol)

#load oneshots data
test_onehots_data=pickle.load(open("../SR-ARE-score/names_onehots.pickle","rb"))
#get "oneshots"
test_onehots=np.array(test_onehots_data["onehots"] )
#reshape test_onehots
test_onehots=test_onehots.reshape(test_onehots.shape[0],70,325,1)

#load model
model =  tf.contrib.keras.models.load_model('my_model.h5')

model.compile(optimizer = 'adam' , loss = 'binary_crossentropy',metrics=['accuracy'])

model.predict(test_onehots)


results = model.predict_classes(test_onehots)
print(results)

with open('labels.txt','w') as f:
   for line in results:
        np.savetxt(f, line, fmt='%i')