#   ...................................................................................
'''
author: Palash Nandi.
'''
#   ...................................................................................

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.linear_model import LogisticRegression

#   ...................................................................................
#

class LR(layers.Layer):
    def __init__(self, dim):
        super(LR,self).__init__()
        self.w = self.add_weight(shape=(dim,1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)

        # print(f'{self.w.shape}: {self.w}')
        # print(f'{self.b.shape}: {self.b}')
    
    def call(self, x):
        raw_value = tf.add( tf.matmul(x,self.w), self.b)
        # print(f'{raw_value.shape}, {raw_value}')
        sigmoid_value = tf.math.sigmoid(raw_value)
        # print(f'{sigmoid_value.shape}, {sigmoid_value}')

        return sigmoid_value

#   ...................................................................................
#
def find_accuracy(dataset):
    total_match = 0
    total_cases = 0

    for step, batch_i in enumerate(dataset):
        x, y_act = batch_i
        res = model(x)
        res = np.where(res > 0.5, 1, 0)
        res = np.transpose(res)[0]
        
        _ = np.logical_xor(y_act, res)

        total_match += np.size(_) - np.count_nonzero(_)
        total_cases += x.shape[0]

        # print(y_act)
        # print(res)
        # print(_)
        # break
    
    # print(total_match)
    # print(total_cases)
   
    return total_match / total_cases



# .......................................................................................
#
path = '/home/palash/ML_GitHub/LogisticRegression/Pima_Indian_Diabetes.csv'
df = pd.read_csv(path)
# print(df.shape)
# print(df.columns)
Y = df['Target'].to_numpy()
X = df.drop(['Target'], axis=1).to_numpy()
# print(df_X.columns)
# print(df_Y.columns)
# print(X.shape)
# print(Y.shape)

total_epoch = 300
batch_size = 64
step_jump = 10
model = LR(X.shape[1])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.BinaryCrossentropy()

train_dataset = tf.data.Dataset.from_tensor_slices((X,Y))
train_dataset = train_dataset.batch(batch_size)


for epoch_i in range(total_epoch):
    loss_epoch = 0
    # print(f'start of epoch {epoch_i}')

    for step, batch_i in enumerate(train_dataset):
        x, y_act = batch_i
        with tf.GradientTape() as tape:
            # print(x)
            # print(y_act)
            res = model(x)
            loss_batch = loss_fn(y_act, res)
            loss_epoch += loss_batch

        grads = tape.gradient(loss_batch, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    if epoch_i % step_jump == 0:
        print(f'LogisticRegression from scratch, {epoch_i}. loss: {loss_epoch}, acc: {find_accuracy(train_dataset)}')
        
    # metric = tf.keras.metrics.BinaryAccuracy()
    # metric.update_state()

    # break



# using sequential() of tf.
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# model = tf.keras.Sequential()
# model.add(LR(X.shape[1]))
# model.compile(optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])
# model.fit(X, Y, epochs=1000, batch_size=32, verbose=1)


#   ...................................................................................
#
print(f'\nNow using LogisticRegression of sklearn.')
lr_sklearn = LogisticRegression(random_state=0, max_iter=total_epoch).fit(X, Y)
score_ = lr_sklearn.score(X, Y)
print(f'using LogisticRegression of sklearn acc: {score_}\n')    