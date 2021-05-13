import  tensorflow as tf
import numpy as np
import pickle


train_symbol_data=np.loadtxt("SR-ARE-train\\names_labels.txt",dtype='str',delimiter=',')
new_symbol = []
#print(train_symbol_data)

#print(train_symbol)

#parese and change to float
for x in train_symbol_data:
    y=np.delete(x, 0)
    new_symbol.append([int(y)])

train_symbol=np.array(new_symbol)
#print(train_symbol)

train_onehots_data=pickle.load(open("SR-ARE-train\\names_onehots.pickle","rb"))


train_onehots=np.array( train_onehots_data["onehots"])

#train_onehots=train_onehots.reshape(train_onehots.shape[0],70,325,1)
#train_onehots=train_onehots.reshape(train_onehots.shape[0],70,325,1)

#Batch and shuffle
train_onehots=np.expand_dims(train_onehots,axis=3)
train_dataset = tf.data.Dataset.from_tensor_slices((train_onehots, train_symbol))
BATCH_SIZE = 8
SHUFFLE_BUFFER_SIZE = 1000
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE,reshuffle_each_iteration=True).batch(BATCH_SIZE)
print(train_dataset)


model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',input_shape=(70,325,1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
#tf.keras.layers.Dropout(0.2)
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#tf.keras.layers.Dropout(0.3)
#model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer = 'adam' , loss = 'binary_crossentropy',metrics=['accuracy'])


class_weight = {0: 0.25,
               1: 0.75}

model.fit(train_dataset ,epochs = 15,class_weight=class_weight)



#model.fit(train_onehots,train_symbol ,epochs = 20)
model.save('my_model.h5')


