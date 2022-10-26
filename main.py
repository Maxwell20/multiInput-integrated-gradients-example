
#import
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, utils
from tensorflow.keras.layers import Activation, Dense, Flatten, Embedding, Dropout, LSTM, Lambda,SimpleRNN, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
from alibi.explainers import IntegratedGradients
import pandas as pd
origional_stdout = sys.stdout
# import regularizer
from tensorflow.keras.regularizers import l1
import random
import csv
import pickle
from sklearn import tree
import sklearn.metrics as skm
from functools import partial
import shap
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import shutil

def preprocess_data(features, labels, model_name):
    # SHUFFLE THE DATA
    features, labels = shuffle(features, labels)

    # Split the data
    X_train, X_rem, Y_train, Y_rem = train_test_split(features,labels, train_size=0.7)
    X_val, X_test, Y_val, Y_test = train_test_split(X_rem, Y_rem, train_size=0.5)
    Y_train = Y_train.reshape(-1,1)
    Y_test = Y_test.reshape(-1,1)
    Y_val = Y_val.reshape(-1,1) 

    # OneHot Encode the labels
    onehot_encoder = OneHotEncoder(sparse=False)
    prepend_zeros = lambda m : np.hstack((np.array([np.zeros(len(m))]).T, m))
    Y_train_onehot = onehot_encoder.fit_transform(Y_train)
    Y_train_onehot = prepend_zeros(Y_train_onehot)
    Y_val_onehot = onehot_encoder.fit_transform(Y_val)
    Y_val_onehot = prepend_zeros(Y_val_onehot)
    Y_test_onehot = onehot_encoder.fit_transform(Y_test)
    Y_test_onehot = prepend_zeros(Y_test_onehot)

    # Save the data
    np.save('test_samples/' + model_name + "_X_test" + '.npy', X_test)
    np.save('test_samples/' + model_name + "_X_train" + '.npy', X_train)
    np.save('test_samples/' + model_name + "_X_val" + '.npy', X_val)
    np.save('test_samples/' + model_name + "_Y_test" + '.npy', Y_test)
    np.save('test_samples/' + model_name + "_Y_train" + '.npy', Y_train)
    np.save('test_samples/' + model_name + "_Y_val" + '.npy', Y_val)
    np.save('test_samples/' + model_name + "_Y_test_onehot" + '.npy', Y_test_onehot)
    np.save('test_samples/' + model_name + "_Y_train_onehot" + '.npy', Y_train_onehot)
    np.save('test_samples/' + model_name + "_Y_val_onehot" + '.npy', Y_val_onehot)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val,Y_train_onehot, Y_test_onehot, Y_val_onehot
    
def plot_importance(feat_imp, feat_names, class_idx, **kwargs):
    """
    Create a horizontal barchart of feature effects, sorted by their magnitude.
    """
    print(feat_imp, feat_names, class_idx)

    df = pd.DataFrame(data=feat_imp, columns=feat_names).sort_values(by=0, axis='columns')
    feat_imp, feat_names = df.values[0], df.columns
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(feat_imp))
    ax.barh(y_pos, feat_imp)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel(f'Feature effects for class {class_idx}', fontsize=15)
    plt.show()
    return ax, fig

def format_for_rnn(data,_input_length):
    #normalize the data
    normalized= tf.keras.utils.normalize(data[:,:3], axis=0, order=2)
    labels = data[:,-1].reshape(1,-1).T
    time = data[:,3].reshape(1,-1).T
    data = np.append(normalized_velocities, normalized, axis = 1)
    data = np.append(data, time, axis = 1)
    data = np.append(data, labels, axis = 1)


    #format the data for LSTM/RNN
    lstm_data = []
    if _input_length != None:
        input_length = _input_length
    else:
        input_length = 3

    lstm_labels = []
    for i in range(len(data)-1):
        if data[i][-2] == 0:#checking the time for a new track
            sample = []         
            sample.append(data[i][:3])
            i = i + 1
            j = i
            while(data[j][-2] != 0 and j < len(data)-1):
                
                if len(sample) < input_length:
                    sample.append(data[j][:3])
                else:
                    lstm_data.append(sample)
                    lstm_labels.append(data[j][-1])
                    sample = []              
                j = j + 1
    lstm__data = np.asarray(lstm_data, dtype=object).astype('float32')
    lstm_labels = np.asarray(lstm_labels)
    return lstm_data, lstm_labels


def build_rnn(_input_length = None):
    input_length = _input_length

    # LOAD THE DATA 
    data = load_data()

    lstm_data = format_for_rnn(data, _input_length)
    lstm_data = lstm_data[0]
    lstm_labels = lstm_data[1]
 
    # Shuffle and split the data 
    X_train, X_test, X_val, Y_train, Y_test, Y_val, Y_train_onehot, Y_test_onehot, Y_val_onehot = preprocess_data(lstm_data, lstm_labels, "RNN")

    ############################# MODEL ##############################################
    inp = Input(shape=(input_length,6), name='input_classification')#input layer
    x = SimpleRNN(64)(inp) # hidden layer
    x = Dropout(.2)(x)
    x = Dense(32, activation = "relu")(x)
    x = Dense(64, activation = "relu")(x)
    x = Dense(128, activation = "relu")(x)
    x = Dense(64, activation = "relu")(x)
    x = Dense(32, activation = "relu")(x)
    z = Dense(18, activation='softmax', name="Classification")(x)# output layer
    model = Model(inputs=inp, outputs=[z])
    opt = Adam(.0001)
    losses = {'Classification': 'categorical_crossentropy',}
    model.compile(loss=losses, optimizer=opt, metrics={'Classification':'accuracy'})
    ############################## END MODEL #######################################

    #show me the model
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True, to_file="RNN_Model.png")
    #training the model
    reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1,
        mode="auto",
        min_delta=0.0001,
        cooldown=2,
        min_lr=0
    )

    history = model.fit(x=X_train, y=Y_train_onehot,
    validation_data=(X_val, Y_val_onehot),
	epochs=75, batch_size=5, callbacks=[reducer],verbose=2)

    #gather metrics for the model
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']   
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs=range(len(acc))
    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.savefig("plots/rnn_accuracy.png")
    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.savefig("plots/rnn_loss.png")
    plt.show()
    #save the model
    model.save("models/rnn/")
    
def test_model():
    #loading the model
    features = ["x", "y", "z"]

    clf = tf.keras.models.load_model('models/rnn/')
    test_data = np.load('test_samples/RNN_X_test.npy')
    test_data_label = np.load('test_samples/RNN_Y_test_onehot.npy')

    # Select a random sample
    index = random.randrange(len(test_data))
    target = next(key for key, value in Target_Dictionary.items() if value == np.argmax(test_data_label[index]))
    print("Label: ", target)
    print(test_data[index])
    test = test_data[index].reshape(1,3,3)
    res = clf.predict(test)
    print("Output:", res)
    target = next(key for key, value in Target_Dictionary.items() if value == np.argmax(res[0]))
    print("Prediction: ", target)

    for t in Target_Dictionary:
        print(t,end =" ")
    print('\n')
    for i in res[0]:
        print(i, end = " ")


    target_fn = partial(np.argmax, axis=1)
    ig  = IntegratedGradients(clf, 
                                layer=None,
                                target_fn=target_fn,
                                n_steps=50,
                                method = "gausslegendre",
                                internal_batch_size=100 )
    predictions = clf(test).numpy().argmax(axis=1)
    explanation = ig.explain(test,
                    baselines=None,
                    target=None)

    attrs_raw = np.array(explanation.data["attributions"]).reshape(3,-1).T
    preds_raw = explanation.data["predictions"]
    plot_importance(list(map(abs, attrs_raw)), features, target)


#entry point of the program
if __name__ == "__main__":
  #call functions here
  test_model()


    
