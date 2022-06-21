import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend

import matplotlib.pyplot as plt
import pickle
from numpy import load

def get_dataset(npz_name):
    # load dataset npz file
    data = load(npz_name)
    x, y = data['arr_0'], data['arr_1']
    
    print(x.shape, y.shape)
    return x, y

def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)

    if backend.image_data_format() == 'channel_first':
        inputShape = (depth, height, width)

    model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=inputShape))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(classes, activation='sigmoid'))
    return model

def summarize_diagnostics(history):
    # plot loss
    plt.subplot(111)
    plt.title('Loss')
    plt.plot(history.history['loss'], color='red', label='train_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()
    
    # plot accuracy
    plt.subplot(111)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train_acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()

    
    
# Load the train data and label from the training data npz file
trainX, trainY = get_dataset('train256_dataset.npz')

# Show the first nine book cover in the dataset
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(trainX[i].reshape(256, 256, 3))

# Build the model
model = build(256, 256, 3, 5)

# Show the model's summary
model.summary()

# Defining the Adam optimizer
opt = Adam(learning_rate=1e-4, decay=1e-6)

# Compile the model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Defining model-checkpoint to save the model every epoch is done
mc = ModelCheckpoint('genre_model.h5',
                      monitor='accuracy',
                      mode='max', 
                      verbose=1,
                      save_best_only=True)

train = model.fit(trainX, trainY, 
                  epochs=20, 
                  verbose=1, 
                  callbacks=[mc])


# If you not defining any model-checkpoint and want to save the model right 
# after the training process, uncomment this line below.
# model.save('genre_model', save_format='h5')

# Show training process' plot
summarize_diagnostics(train)

# Save training log history
with open('train_history', 'wb') as file_pi:
    pickle.dump(train.history, file_pi)

train_history = pickle.load(open('train_history', 'rb'))
