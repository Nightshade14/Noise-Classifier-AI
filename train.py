# Importing basic libraries
import os
import librosa
import librosa.display
import glob
import skimage
import pickle
from tqdm.auto import trange, tqdm

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

basePath = "/kaggle/input/urbansound8k"

meta_data_file_path = basePath + "/UrbanSound8K.csv"
df = pd.read_csv(meta_data_file_path)

feature = []
label = []

def parser(df):
    '''

    Function to load files and extract features using melspectogram

    '''

    for i in trange(df.shape[0]):
        file_name = basePath + '/fold' + str(df["fold"][i]) + '/' + df["slice_file_name"][i]
        X, sample_rate = librosa.load(file_name)
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate),axis=1)
        feature.append(mels)
        label.append(df["classID"][i])
    return [feature, label]


mel_features = parser(df)

X_root = np.array(mel_features[0])
y_root = np.array(mel_features[1])

X = X_root.copy()
y = y_root.copy()


# Initializing variables
random_state = 76
test_size = 0.25
is_shuffled = True
is_stratified = y


standard_scaling = True
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=is_shuffled, stratify=is_stratified)
if standard_scaling == True:
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)
    

# Training Logisitic Regression model with l2 penalty
from sklearn.linear_model import LogisticRegression
log_reg_clf = LogisticRegression(penalty="l2", max_iter=10000, random_state=random_state)
log_reg_clf.fit(X_train, y_train)
print(f"Number of iterations to converge: {log_reg_clf.n_iter_[0]}")
print(log_reg_clf.score(X_test, y_test))



# SVC

standard_scaling = False
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=is_shuffled, stratify=is_stratified)
if standard_scaling == True:
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)
    
from sklearn.svm import SVC
svc_clf = SVC(C=1e3, kernel="rbf", gamma="auto", random_state=random_state)
svc_clf.fit(X_train, y_train)
print(svc_clf.score(X_test, y_test))


# Decision Tree

standard_scaling = False
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
if standard_scaling == True:
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)
    
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None, ccp_alpha=0.0001)
dt_clf.fit(X_train, y_train)
print(dt_clf.score(X_test, y_test))


# Random Forest

standard_scaling = True
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=is_shuffled, stratify=is_stratified)
if standard_scaling == True:
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)
    
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(criterion="gini", max_depth=None, ccp_alpha=0.0001, random_state=random_state)
rf_clf.fit(X_train, y_train)
print(rf_clf.score(X_test, y_test))




# XgBoost

standard_scaling = False
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=is_shuffled, stratify=is_stratified)
if standard_scaling == True:
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)
    
import xgboost
xgb_clf = xgboost.XGBClassifier(max_depth=None, objective="multi:softprob",random_state=random_state)
xgb_clf.fit(X_train, y_train)
print(xgb_clf.score(X_test, y_test))


# ANN

standard_scaling = False
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=is_shuffled, stratify=is_stratified)
if standard_scaling == True:
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)
    
import tensorflow as tf
import keras
from keras import layers


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)
        

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    activation_fn = "relu"
    model = keras.Sequential(
        [
            layers.Dense(units=128, activation=activation_fn, input_shape=(128,)),
            layers.Dense(units=256, activation=activation_fn),
            layers.Dense(units=512, activation=activation_fn),
            layers.Dropout(0.2),
            layers.Dense(units=337, activation=activation_fn),
            layers.Dense(units=221, activation=activation_fn),
            layers.BatchNormalization(),
            layers.Dense(units=164, activation=activation_fn),
            layers.Dense(units=88, activation=activation_fn),
            layers.Dense(units=41, activation=activation_fn),
            layers.BatchNormalization(),
            layers.Dense(units=23, activation=activation_fn),
            layers.Dense(units=10, activation="Softmax")
        ]
    )


    es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=10e-7,
    patience=15,
    verbose=1,
    mode='min',
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=100
)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 metrics = ['accuracy']
                 )

#     We are not going to use the Early Stopping callback in order to visualize the plateau of the training curves.

    history = model.fit(X_train, y_train, batch_size=512, epochs=500, validation_data=(X_test, y_test))






# CNN


    standard_scaling = False

X_train_CNN, X_test_CNN, y_train_CNN, y_test_CNN = train_test_split(X, y, test_size=0.15, random_state=random_state, shuffle=is_shuffled, stratify=is_stratified)
if standard_scaling == True:
    std_scaler = StandardScaler()
    X_train_CNN = std_scaler.fit_transform(X_train_CNN)
    X_test_CNN = std_scaler.transform(X_test_CNN)

X_train_CNN = X_train_CNN.reshape(X_train_CNN.shape[0], 16, 8, 1)
X_test_CNN = X_test_CNN.reshape(X_test_CNN.shape[0], 16, 8, 1)
input_dim = (16, 8, 1)

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    model_CNN = keras.Sequential(
        [
            layers.Conv2D(128, (3, 3), padding = "same", activation = "tanh", input_shape = input_dim),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Conv2D(64, (3, 3), padding = "same", activation = "tanh"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Dropout(0.21),
            layers.Conv2D(32, (3, 3), padding = "same", activation = "tanh"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(1024, activation = "tanh"),
            layers.Dense(10, activation = "softmax")
        ]
    )


    es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=5,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=5
)

#     Adam is efficient with large datasets, high dimensional spaces and is robust to noisy gradients
#     Sparse Categorical Cross-entropy loss is memory efficient and suitable for our label type.

    model_CNN.compile(optimizer="adam",
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 metrics = ['accuracy']
                 )

#     We are not going to use the Early Stopping callback in order to visualize the plateau of the training curves.

    history_CNN = model_CNN.fit(X_train_CNN, y_train_CNN, batch_size=256, epochs=500, validation_data=(X_test_CNN, y_test_CNN), validation_batch_size=256)




