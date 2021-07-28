##############################Load dataset#####################################
import pandas as pd
data = pd.read_csv('dataset_DL_1.csv')

##############################Memilih kolom data dari file CSV#################
X = data.iloc[:,0:16].values
y = data.iloc[:, 16].values

# Pre-Processing
##############################Mengubah data teks menjadi angka vektor##########
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
y_vector = cv.fit_transform(y).toarray()

##############################Mengubah rentang nilai vektor menjadi 0 dan 1####
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
y_normalized = min_max_scaler.fit_transform(y_vector)

##############################Memberikan Label pada Dataset####################
from sklearn.preprocessing import label_binarize
y_binarized = label_binarize(y_normalized, classes=[0, 1, 2, 3, 4])
n_classes = y_normalized.shape[1]

##############################Membagi dataset##################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, 
                                                    test_size = 0.2, 
                                                    random_state = 1)

##############################Membangun model deep learning####################
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import numpy as np
import numpy
input_dim = 16
output_dim = 16
X_train = sequence.pad_sequences(X_train, maxlen=output_dim)
X_test = sequence.pad_sequences(X_test, maxlen=output_dim)
embedding_vector_length = 16
model = Sequential()
model.add(Embedding(input_dim, output_dim, input_length=output_dim))
model.add(LSTM(350))
model.add(Dense(5, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

##############################Training model deep learning#####################
history = model.fit(X_train, y_train, epochs=20, batch_size=128, verbose=0, 
                    shuffle=False, validation_data=(X_test, y_test))

##############################Membuat prediksi model deep learning#############
data_original = pd.DataFrame(y_test)
data_predictions = pd.DataFrame(model.predict(X_test))

##############################Normalisasi hasil prediksi#######################
data_original = min_max_scaler.fit_transform(data_original)
data_predictions = min_max_scaler.fit_transform(data_predictions)

##############################Ubah data jadi nilai bulat#######################
x_olah = np.round(data_original)
y_olah = np.round(data_predictions)
vector = np.vectorize(np.int)
x_olah = vector(x_olah)
y_olah = vector(y_olah)
original_fix = pd.DataFrame(x_olah)
predicted_fix = pd.DataFrame(y_olah)

##############################Plot hasil prediksi##############################
import seaborn as sns
sns.set_theme(style="darkgrid")
ax = sns.lineplot(x=original_fix.index, y=original_fix[0], label="Data original", 
                  color='green')
ax = sns.lineplot(x=predicted_fix.index, y=predicted_fix[0], label="Data Prediksi", 
                  color='red')
ax.set_title('Kerusakan Conveyor', size = 14, fontweight='bold')
ax.set_xlabel("Hari", size = 10, fontweight='bold')
ax.set_ylabel("Kerusakan", size = 10, fontweight='bold')
ax.set_xticklabels('', size=10)
    

##############################Evaluasi model Deep Learning#####################

##############################Akurasi##########################################

##############################Menghitung nilai akurasi#########################
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy from LSTM: %.2f%%\n" % (scores[1]*100))

##############################Menghitung nilai akurasi dengan metode KFold#####
seed = 7
numpy.random.seed(seed)

from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
results = []
names = []
for train, test in kfold.split(X, y_normalized):
  # create model

    model = Sequential()
    model.add(Embedding(input_dim, output_dim, input_length=output_dim))
    model.add(LSTM(350))
    model.add(Dense(5, activation='sigmoid'))
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
    model.fit(X[train], y_normalized[train], epochs=20, batch_size=128, verbose=0,
              shuffle=False, validation_data=(X_test, y_test))
	# evaluate the model
    scores = model.evaluate(X[test], y_normalized[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

##############################Plot Data Accuracy dan Loss######################
history_dict = history.history
history_dict.keys()
acc = history_dict['accuracy']
loss = history_dict['loss']
val_acc = history_dict['val_accuracy']
val_loss = history_dict['val_loss']
epochs = range(len(loss))
import matplotlib.pyplot as plt
plt.figure()
plt.plot(epochs, loss, 'r', label='Data Training Loss')
plt.plot(epochs, acc, 'g', label='Data Training Accuracy')
plt.plot(epochs, val_loss, 'b', label='Data Testing Loss')
plt.plot(epochs, val_acc, 'y', label='Data Testing Accuracy')
plt.title("Training dan Accuracy Loss")
plt.legend(loc="right")
plt.show()



##############################Confusion Matrix#################################
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

##############################Menghitung Confusion Matrix######################
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
data_asli = original_fix.values.tolist()
data_prediksi = predicted_fix.values.tolist()
data_asli=np.argmax(data_asli, axis=1)
data_prediksi=np.argmax(data_prediksi, axis=1)
cm = confusion_matrix(data_asli, data_prediksi)
cm = numpy.flip(cm)
from sklearn.metrics import accuracy_score, precision_score, recall_score
Accuracy_confusion = accuracy_score(data_asli, data_prediksi)
Precision_confusion = precision_score(data_asli, data_prediksi, average='macro')
Recall_confusion = recall_score(data_asli,data_prediksi, average='macro')
print("Accuracy of Confusion: %.2f%%" % (Accuracy_confusion*100))
print("Precision of Confusion: %.2f%%" % (Precision_confusion*100))
print("Recall of Confusion: %.2f%%\n" % (Recall_confusion*100))

##############################Plot Confusion Matrix############################
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(cm, annot=True, fmt="1.0f", cmap='Blues')
plt.xlabel("predicted")
plt.ylabel("truth")
plt.show()



##############################ROC##############################################
##############################Menghitung nilai ROC#############################
from sklearn.metrics import roc_auc_score
roc_score = roc_auc_score(data_original, data_predictions,multi_class='ovo', 
                          average="macro")
print("ROC Score from LSTM = %.2f%%" % (roc_score*100))

##############################Plot grafik ROC##################################
from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], data_predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
fpr["nano"], tpr["nano"], _ = roc_curve(y_test.ravel(), 
                                        data_predictions.ravel())
roc_auc["nano"] = auc(fpr["nano"], tpr["nano"])

import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='Predicted (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, label='Predicted', 
         linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Prediction')
plt.ylabel('True Data')
plt.title('ROC')
plt.show()





###############################################################################
##############################USER INTERFACE###################################


##############################Input level dan rekomendasi ke dataframe#########
pre_prediksi = np.array(y_olah).astype('str')
pre_prediksi = pd.DataFrame(pre_prediksi,columns=['Berat','Berbahaya','Menengah',
                                                  'Normal','Ringan'])
pre_prediksi["level"] = pre_prediksi["Berat"] + pre_prediksi["Berbahaya"] + pre_prediksi["Menengah"] + pre_prediksi["Normal"] + pre_prediksi["Ringan"]
pre_prediksi['level'] = pre_prediksi['level'].replace(
    ['10000','01000','00100','00010','00001','00101','11000','10100'],
    ['BERAT','BERBAHAYA','MENENGAH','NORMAL','RINGAN','MENENGAH','BERBAHAYA','BERAT'])
pre_prediksi["perawatan"] = pre_prediksi["Berat"] + pre_prediksi["Berbahaya"] + pre_prediksi["Menengah"] + pre_prediksi["Normal"] + pre_prediksi["Ringan"]
pre_prediksi['perawatan'] = pre_prediksi['perawatan'].replace(
    ['10000','01000','00100','00010','00001','00101','11000','10100'],
    ['Cek conveyor, ekstra hati-hati',
     'Hentikan penggunaan conveyor, Cek conveyor keseluruhan',
     'Mulai cek conveyor, mulai ekstra hati-hati',
     'Selalu waspada',
     'Mulai waspada, lapor ke teknisi',
     'Mulai cek conveyor, mulai ekstra hati-hati',
     'Hentikan penggunaan conveyor, Cek conveyor keseluruhan',
     'Cek conveyor, ekstra hati-hati'])
data_prediksi = pre_prediksi["level"].tolist()
data_rekomendasi  = pre_prediksi["perawatan"].tolist()



from tkinter import Label, Button, Tk
##############################inisiasi User Interface##########################
root = Tk()

##############################Ukuran dan judul UI##############################
root.geometry('640x480')
root.title("Program Prediksi Kerusakan Mesin Conveyor by PENS")

##############################Menampilkan hari dan jam secara realtime#########
from datetime import datetime
def tick():
    now = datetime.now().strftime('%d-%m-%y %H:%M:%S')
    clock.config(text=now)
    clock.after(200, tick)
clock = Label(root, font='ariel 10', bg="white", fg="black")
clock.grid(row=0, column=0)
clock.place(x=530,y=0)
tick()

##############################Selamat datang###################################
myLabel = Label(root, text="Selamat datang di Prediksi Kerusakan Mesin Conveyor", 
                fg="black", font='times 12')
myLabel.grid(row=6, column=15)
myLabel.place(x=165, y=39)
myLabel1 = Label(root, text="Program ini digunakan untuk memprediksi kerusakan mesin conveyor", 
                 fg="black",font='times 12')
myLabel1.grid(row=8, column=15)
myLabel1.place(x=120, y=60)


##############################Hasil Prediksi###################################
day = datetime.now().strftime('%d')
a = int(day)

kemarin = a - 1
hari_ini = a
besok = a + 1

Kemarin = Label(root, text="Kemarin ada kerusakan %s di salah satu conveyor" 
                       % data_prediksi[kemarin], fg="black", font='ariel 12' )
Kemarin.grid(row=10, column=15)
Kemarin.place(x=100, y=200)

HasilKerusakan = Label(root, text="Hari ini diprediksi ada kerusakan %s di salah satu conveyor" 
                       % data_prediksi[hari_ini], fg="black", font='ariel 12' )
HasilKerusakan.grid(row=12, column=15)
HasilKerusakan.place(x=100, y=250)
RekomendasiAksi = Label(root, text="Rekomendasi: %s" 
                       % data_rekomendasi[hari_ini], fg="black", font='ariel 12' )
RekomendasiAksi.grid(row=14, column=15)
RekomendasiAksi.place(x=100, y=300)

MasaDepan = Label(root, text="Besok diprediksi ada kerusakan %s di salah satu conveyor" 
                       % data_prediksi[besok], fg="black", font='ariel 12' )
MasaDepan.grid(row=16, column=15)
MasaDepan.place(x=100, y=350)

##############################Fungsi Close Button##############################
def clicked():
    root.destroy()   
btn = Button(root, text = "Close" ,
             fg = "red", command=clicked)
btn.grid(column=2, row=0)
btn.place(x=540,y=430)

##############################Jalankan program#################################
root.mainloop()