#	...................................................................................
'''
author: Palash Nandi.
'''
#	...................................................................................
import math
import numpy as np
import pandas as pd
from sklearn import datasets
from pprint import pprint
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 

# Logistic Regression 
#
class LR(layers.Layer):
    def __init__(self, dim):
        super(LR,self).__init__()
        self.w = self.add_weight(shape=(dim,1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)

        
        # print(f'weights.shape: {np.array(self.w).shape}')
        # print(f'bias.shape: {np.array(self.b).shape}')
    
    def call(self, x):
        raw_value = tf.add( tf.matmul(x,self.w), self.b)
        # print(f'{raw_value.shape}, {raw_value}')
        sigmoid_value = tf.math.sigmoid(raw_value)
        # print(f'{sigmoid_value.shape}, {sigmoid_value}')

        return sigmoid_value

def find_accuracy(model, dataset):
    total_match = 0
    total_cases = 0

    for step, batch_i in enumerate(dataset):
        x, y_act = batch_i
        
        # print(f'x.shape in acc(): {x.shape}')
        # print(f'y.shape in acc(): {y_act.shape}')

        res = model(x)
        res = np.where(res > 0.5, 1, 0)
        # res = np.transpose(res)[0]
        
        _ = np.logical_xor(y_act, res)

        total_match += np.size(_) - np.count_nonzero(_)
        total_cases += x.shape[0]
        
        # print(f'y_act.shape: {y_act.shape}')
        # print(f'res.shape: {res.shape}')
        # print(f'total_cases: {total_cases}')
        # print(_.shape)
        # break
    
    # print(total_match)
    # print(total_cases)
    result_dict = {'total_cases':total_cases, 'total_match':total_match, 'accuracy':total_match/total_cases}
    return  result_dict

def train_model(dict_):
    total_epoch = 300
    step_jump = 10

    model = dict_['model']
    optimizer = dict_['optimizer']
    loss_fn = dict_['loss_fn']
    train_dataset = dict_['train_dataset']
    test_dataset = dict_['test_dataset']


    for epoch_i in range(total_epoch):
        loss_epoch = 0
        # print(f'start of epoch {epoch_i}')

        for step, batch_i in enumerate(train_dataset):
            x, y_act = batch_i
            # print(f'x: {x.shape}')
            # print(f'y: {y_act.shape}')

            with tf.GradientTape() as tape:
                # print(x)
                # print(y_act)
                res = model(x)
                loss_batch = loss_fn(y_act, res)
                loss_epoch += loss_batch

            grads = tape.gradient(loss_batch, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if epoch_i % step_jump == 0:
            
            result_dict = find_accuracy(model, test_dataset)
            print(f"\t{epoch_i}. loss: {loss_epoch}, acc: {result_dict['accuracy']}")
            
        # print(model.weights)
        # print(np.array(model.weights).shape)
        # break
    
def full_scale_evaluation(dataset):
    total_cases = 0
    total_matched = 0

    for step, batch_i in enumerate(dataset):
        flag = 0
        scores = None
        x, y_act = batch_i
        
        for class_i in unique_classes:
            model_dict = class_param_dict[class_i]
            model = model_dict['model']
            result = model(x)
            
            if flag == 0:
                scores = result.numpy()
                flag = 1
            else:
                scores = np.concatenate((scores, result), axis=1)

            

            # break
        # print(scores.shape)
        
        pred_class = scores.argmax(axis=1)
        for i, p_cls in enumerate(pred_class):
            # print(f'p_cls: {p_cls}, y_act[i]: {y_act[i]} =< {p_cls == y_act[i]}')
            if p_cls == y_act[i]:
                total_matched += 1

        total_cases += x.shape[0]
        # print(f'total_matched: {total_matched}\ttotal_cases: {total_cases}')


        # break
    print(f'acc: {total_matched/total_cases}')
            






# load and scale the data
#
min_max_scaler = MinMaxScaler()
digits = datasets.load_digits( n_class= 10 , return_X_y= True )	
X_raw = pd.DataFrame(digits[0])
# print( X_raw.head())
col_names = X_raw.columns
X_raw[ col_names ] = min_max_scaler.fit_transform( X_raw[ col_names ])
# print( X_raw.head())
Y = pd.DataFrame( digits[1], columns=['target'] )
unique_classes = set(Y['target'].values)
total_target_class = len(unique_classes)

# print(f'X_raw.shape: {X_raw.shape}')
# print(f'Y.shape: {Y.shape}')
# print( Y.head())
# print(f'unique_classes: {unique_classes} => {total_target_class}')

# train & test split
#
tot_data = X_raw.shape[0]
tot_cols = X_raw.shape[1]
tr_data = math.ceil( tot_data * .8 )

print(f'Total training data: {tr_data}')
print(f'Total test data: { tot_data - tr_data }')

train_X = X_raw.values[ :tr_data, :]
test_X  = X_raw.values[ tr_data:, :]
train_Y = Y.values[ :tr_data, :]
test_Y  = Y.values[ tr_data:, :]

print(f'train_X.shape: {train_X.shape}, train_Y.shape: {train_Y.shape}')
print(f'test_X.shape: {test_X.shape}, test_Y.shape: {test_Y.shape}')

# Training start
total_epoch = 1
batch_size = 64

class_param_dict = {}

for class_i in unique_classes:
    print(f'For class_{class_i}')
    temp_train_Y = np.where(train_Y==class_i,1,0)
    temp_test_Y = np.where(test_Y==class_i,1,0) 

    # print(f'temp_train_Y: {temp_train_Y.shape}')
    # print(f'temp_test_Y : {temp_test_Y.shape}')
    # print(test_Y.shape)
    # print(temp_test_Y.shape)


    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, temp_train_Y)).batch(batch_size)
    test_dataset  = tf.data.Dataset.from_tensor_slices((test_X, temp_test_Y)).batch(batch_size)

    dict_ = {'model':None, 'optimizer':None, 'loss_fn':None}
    
    dict_['model'] = LR(train_X.shape[1])
    dict_['optimizer'] = tf.keras.optimizers.Adam(learning_rate=1e-3)
    dict_['loss_fn'] = tf.keras.losses.BinaryCrossentropy()
    dict_['train_dataset'] = train_dataset
    dict_['test_dataset'] = test_dataset

    class_param_dict[class_i] = dict_
    
    # for step, batch_i in enumerate(train_dataset):
    #     x, y_act = batch_i
    #     print(f'x.shape: {x.shape}, y_act.shape: {y_act.shape}')

    # for step, batch_i in enumerate(test_dataset):
    #     x, y_act = batch_i
    #     print(f'x.shape: {x.shape}, y_act.shape: {y_act.shape}')
        
    train_model(dict_)
    
    # break



test_dataset  = tf.data.Dataset.from_tensor_slices((test_X, test_Y)).batch(batch_size)
full_scale_evaluation(test_dataset)    
