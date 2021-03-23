import json
import numpy as np
import tensorflow.keras as keras


data_path = "/Users/talen/Desktop/Audio_features.json"

#Loading the desired data from the json file
def data_loading(data_path, session_num):
    #Read data from the json file
    with open(data_path, "r") as file:
        data = json.load(file)

    #Specify the data (X) used for predicting and the data (Y) as the target
    X = np.array(data[session_num]["Log-Mel-spectrogram"])
    Y = np.array(data[session_num]["labels"])

    return X,Y


#%% Create the training and testing sets
    #The session 1~4 are used as training, whereas the session 5 is used as testing
sessions_training = ["1","2","3","4"]
session_testing = ["5"]

    #Get the training data including the predictors and targets
X_train_1, Y_train_1 = data_loading(data_path,sessions_training[0])
X_train_2, Y_train_2 = data_loading(data_path,sessions_training[1])
X_train_3, Y_train_3 = data_loading(data_path,sessions_training[2])
X_train_4, Y_train_4 = data_loading(data_path,sessions_training[3])

X_train = np.concatenate((X_train_1, X_train_2, X_train_3, X_train_4))
Y_train = np.concatenate((Y_train_1, Y_train_2, Y_train_3, Y_train_4))

    #Get the testing data including the predictors and targets
X_test, Y_test = data_loading(data_path, session_testing[0])

    #Add an extra dimension — depth to the training and testing data, since the CNN requires 3D data for training
# X_train = X_train[..., np.newaxis]  #4d-array -> [n_data_samples, 47 (time_bins), 40 (log-mel), 1 (depth)]
# X_test = X_test[..., np.newaxis]

#%% Create CNN for extracting features from log-mel-spectrograms
    #Build the CNN
def build_CNN(input_shape):
    #Initiate the model
    model = keras.Sequential()

    #1st conv layer
    model.add(keras.layers.Conv2D(filters=30, kernel_size=(5,5), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2), padding='same'))
        #Use batchnormalisation process to normalise the activations in the current layer and to speed up the training
    model.add(keras.layers.BatchNormalization())

    #2nd conv layer
    model.add(keras.layers.Conv2D(filters=30, kernel_size=(3,3), activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2), padding='same'))
        #Use batchnormalisation process to normalise the activations in the current layer and to speed up the training
    model.add(keras.layers.BatchNormalization())


    #FC layer and dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=32, activation="sigmoid"))
    model.add(keras.layers.Dropout(0.3))

    #output layer — softmax
    model.add(keras.layers.Dense(units=4, activation="softmax"))

    return model

#%%
#The input data has three dimensions, the last two are the time and log-mel-spectrogram coefficient that could be
#taken as the input size. These inputs also have the depth of 1 in input size
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
cnn = build_CNN(input_shape)

    #Compile the model
optimiser = keras.optimizers.Adam(learning_rate=0.0001)
cnn.compile(optimizer=optimiser, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    #Train the model
cnn.fit(x=X_train, y=Y_train, batch_size=32, epochs=30, verbose=1)

    #Evaluate the model
error, accuracy = cnn.evaluate(X_test, Y_test, verbose=1)
print("The accuracy of the moel for SER is: ", accuracy)



#%%
    #Make predictions on a new sample
def predict(model, X, y):
    X=X[np.newaxis, ...]

    #prediction = [[0.1, 0.2, 0.3, ...]]
    prediction = model.predict(X)

    #Extract the index with max value
    predicted_index = np.argmax(prediction, axis=1) #e.g.,[3]
    print("Expected index: {}, Predictied inex: {}".format(y, predicted_index))


X =  X_test[100]
y = Y_test[100]

predict(cnn, X, y)

#%%
import librosa.display
import matplotlib.pyplot as plt

with open(data_path, "r") as file1:
    data1 = json.load(file1)

#1.Read the data from session
s5_labels = data1["5"]["labels"]
s5_specs = data1["5"]["Log-Mel-spectrogram"]

#%% Temporarily use for generating spectrograms
#Read the spectrogram coefficients of session 5
for i, v in enumerate(s5_labels):
    #Read the spectrogram coefficients of ang
    if v == 2:
        spectrogram = s5_specs[i]
        librosa.display.specshow(np.array(spectrogram), x_axis="time", y_axis="log")
        plt.savefig("/Users/talen/Desktop/Temp/"+str(i))
        print("The number {} spectrogram is generated!".format(str(i)))

print("Done!!!")