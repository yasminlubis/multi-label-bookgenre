from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# # The difference between batch and non-batch function is that the batch function use to
# predict and ploting batch, which the non-batch is only for one data predict and plot.

# # So  the predict and ploting batch have a batch output, based on the data's total in a folder.
# # And you also have to adjust the rows and cols number, based on the data's total in that folder.

def plot_image(img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

def plot_value_array(predictions_array):
    plt.grid(False)
    plt.yticks([])
    thisplot = plt.bar(range(5), predictions_array, color="blue")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('blue')
    
    i=0
    for (t, p) in zip(thisplot, predictions_array):
        width = t.get_width()
        height = t.get_height()
        x, y = t.get_xy()
        plt.text(x+width/2,
                 y+height*1.01,
                 str('{:.2f}%'.format(p * 100)),
                 ha='center',)
        i+=1
    
    class_names = ['F', 'G', 'H', 'M', 'R']
    _ = plt.xticks(range(5), class_names)
    
def predict_batch(direktori):
    pred_full = []
    img_full = []

    for file in os.listdir(direktori):
        image = Image.open(direktori+file).convert('RGB')
        image = image.resize((256, 256))
        image = np.asarray(image)
        img_full.append(image)
        image = np.expand_dims(image, axis=0)

        pred = model.predict(image)[0]
        idxs = np.argsort(pred)[::-1][:2]
        pred_full.append(pred)
    
    return img_full, pred_full
  
def plot_batch(img_full, pred_full, rows, cols):
    num_rows = rows
    num_cols = cols
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(img_full[i])
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(pred_full[i])
    plt.tight_layout()
    plt.show()
    
# Load the pre-trained model
model = load_model('256_model.h5')

# Define the folder address, also the rows and cols number
dir_path = 'that_folder/'
img_batch, pred_batch = predict_batch(dir_path)
plot_batch(img_batch, pred_batch, 25, 2)
