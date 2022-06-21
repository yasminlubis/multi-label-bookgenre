import os
import random
import pickle

from imutils import paths
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelBinarizer
from numpy import asarray, savez_compressed, load
from pandas import read_csv

# to load train dataset both raw label and cover image
def load_train_data(folder):
    data = []
    label = []
    
    for file in folder:
        image = load_img(file)
        image = img_to_array(image, dtype='uint8')
        data.append(image)
        
        # split file-path to get each genre folder name
        lb = file.split(os.path.sep)[1].split()
        label.append(lb)
    
    dataset = asarray(data, dtype='uint8')
    labelset = asarray(label)
    labelset = load_train_label(labelset)
    
    return dataset, labelset
  
# to load train data's label
# LabelBinarizer to transform raw genre label from data folder to 1 and 0 value
def load_train_label(label):
    lb = LabelBinarizer()
    label = lb.fit_transform(label)
    
    for (i, tag) in enumerate(lb.classes_):
        print('{}. {}'.format(i + 1, tag))
    
    # save the label binarizer class to device
    pickleSave = open('lb_binarizer', 'wb')
    pickleSave.write(pickle.dumps(lb))
    pickleSave.close()
    
    return label
  
# to load test dataset from folder and test-csv file
def load_test_data(folder, csv):
    data = []

    df = read_csv('test_label.csv')

    for isbn in df['isbn']:
        if isbn in os.listdir(folder):
            file = folder+isbn
            image = load_img(file)
            image = img_to_array(image, dtype='uint8')
            data.append(image)

    dataset = asarray(data, dtype='uint8')
    genre = asarray(df.drop(['isbn'], axis=1))
    
    return dataset, genre
  
# save train image data and label to npz format
def save_train_npz(folder, npz_name):
    list_dir = sorted(list(paths.list_images(folder)))
    random.shuffle(list_dir)
    
    x, y = load_train_data(list_dir)
    print(x.shape, y.shape)
    
    savez_compressed(npz_name, x, y)
    
# save test image data and label to npz format
def save_test_npz(folder, csv, npz_name):
    x, y = load_test_data(folder, csv)
    print(x.shape, y.shape)
    
    savez_compressed(npz_name, x, y)
    
def main():
    # Training Dataset
    train_folder = 'train'
    save_train_npz(train_folder, 'ds_train.npz')
    
    # Testing Dataset
    test_folder = 'test/'
    save_test_npz(test_folder, 'test_label.csv', 'ds_test.npz')
    
main()
