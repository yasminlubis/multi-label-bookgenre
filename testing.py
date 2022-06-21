import pickle
import matplotlib.pyplot as plt
import numpy as np

from numpy import load
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.metrics import f1_score, hamming_loss, accuracy_score

def get_dataset(npz_name):
    data = load(npz_name)
    x, y = data['arr_0'], data['arr_1']
    
    print(x.shape, y.shape)
    return x, y
  
# load test dataset
testX, testY = get_dataset('ds_test.npz')

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(testX[i].reshape(256, 256, 3))

model = load_model('256_model.h5')

pred = model.predict(testX, verbose=1)

y_pred=[]
for sample in  pred:
    y_pred.append([1 if i>=0.2 else 0 for i in sample ] )
y_pred = np.array(y_pred)

print('Actual Genre: \n', testY,
     '\nPredicted Genre: \n', y_pred)

# Macro Average F1 Score
f1_macro = f1_score(testY, y_pred, average='macro')
print("Macro F1 Score: %0.2f%% " % (f1_macro * 100))

# Micro Avrage F1 Score
f1_micro = f1_score(testY, y_pred, average='micro')
print("Micro F1 Score: %0.2f%% " % (f1_micro * 100))

# Hamming Loss value, the loss should be as low as possible and the range is from 0 to 1
print("Hamming Loss: %0.2f%% " % (hamming_loss(testY, y_pred)* 100))

# Confusion Matrix for Multi Label Classification
conf_matrix = multilabel_confusion_matrix(testY, y_pred)
conf_matrix

genre_label = ['Fantasy', 'General', 'Horror', 'Mystery', 'Romance']

# Show the classification report
print(classification_report(testY, y_pred, zero_division=0, target_names=genre_label))




# # # If you guys need the manual way to calculate the metric
# - Precision
# - Recall
# - F1Measure
# - Confusion Matrix

def precision_label(tp, fp):
    precision = tp / (fp + tp)
    return precision

def recall_label(tp, fn):
    recall = tp / (fn + tp)
    return recall

def f1_measure(precision, recall):
    f1 = 2 * ((precision * recall)/(precision + recall))
    return f1

def confusion_matrix_manual(y_true, y_pred):
    tp_fan = tp_gen = tp_hor = tp_mys = tp_rom = 0
    fp_fan = fp_gen = fp_hor = fp_mys = fp_rom = 0
    fn_fan = fn_gen = fn_hor = fn_mys = fn_rom = 0
    tn_fan = tn_gen = tn_hor = tn_mys = tn_rom = 0
    
    tp = [tp_fan, tp_gen, tp_hor, tp_mys, tp_rom]
    fp = [fp_fan, fp_gen, fp_hor, fp_mys, fp_rom]
    fn = [fn_fan, fn_gen, fn_hor, fn_mys, fn_rom]
    tn = [tn_fan, tn_gen, tn_hor, tn_mys, tn_rom]
    
    for (yt, yp) in zip(y_true, y_pred):  # yt = [1,0,0,0,1]   yp = [0,0,0,0,1]
        for i in range(0,5):
            if yt[i]==1 and yp[i]==1:
                tp[i] +=  1
            elif yt[i]==1 and yp[i]==0:
                fn[i] += 1
            elif yt[i]==0 and yp[i]==1:
                fp[i] +=  1
            elif yt[i]==0 and yp[i]==0:
                tn[i] += 1
                
    print('TP :', tp)
    print('FP :', fp)
    print('FN :', fn)
    print('TN :', tn)
    
    return tp, fp, fn, tn

def evaluation_manual(label, tp, fp, fn, tn):
    for (i, j) in enumerate(label):
        p = precision_label(tp[i], fp[i])
        r = recall_label(tp[i], fn[i])
        f1 = f1_measure(p, r)
        
        print('Precision label-%s : %0.2f%%' %(label[i], p * 100))
        print('Recall label-%s : %0.2f%%' %(label[i], r * 100))
        print('F1Score label-%s : %0.2f%% \n' %(label[i], f1 * 100))

tp, fp, fn, tn = confusion_matrix_manual(testY, y_pred)

evaluation_manual(genre_label, tp, fp, fn, tn)
