##############################Load dataset#####################################
from pandas import read_csv
url = "dataset_ML_1.csv"
names = ['cbp','lc','snsr','mtr','ke','js','dc','ls','la','pk','tjpk','mp',
         'grc','splj','kc','rc','level']
dataset = read_csv(url, names=names)

##############################Memilah dan split dataset########################
from sklearn.model_selection import train_test_split
X = dataset.iloc[:,0:16].values
y = dataset.iloc[:, 16].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=1)

###############################Training Machine Learning#######################
from sklearn.svm import SVC
models = []
models.append(('SVM', SVC(gamma='auto')))
model = SVC(decision_function_shape='ovo', probability=True)
model.fit(X_train, y_train)

###############################Memprediksi Kerusakan###########################
predictions = model.predict(X_test)

#Plot
from sklearn.preprocessing import LabelBinarizer
MultiLabel = LabelBinarizer()
y_plot = MultiLabel.fit_transform(y_test)
predictions_plot = MultiLabel.fit_transform(predictions)

import pandas as pd
y_plot = pd.DataFrame(y_plot)
predictions_plot1 = pd.DataFrame(predictions_plot)

import seaborn as sns
sns.set_theme(style="darkgrid")
ax = sns.lineplot(x=y_plot.index, y=y_plot[0], label="Data original", 
                  color='green')
ax = sns.lineplot(x=predictions_plot1.index, y=predictions_plot1[0], 
                  label="Data Prediksi", color='red')
ax.set_title('Kerusakan Conveyor', size = 14, fontweight='bold')
ax.set_xlabel("Hari", size = 10, fontweight='bold')
ax.set_ylabel("Kerusakan", size = 10, fontweight='bold')
ax.set_xticklabels('', size=10)

##############################Evaluasi Prediksi################################
from sklearn.metrics import accuracy_score
svm_accuracy = accuracy_score(y_test, predictions)
print("Accuracy from SVM: %.2f%%" % (svm_accuracy*100))

results = []
names = []
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
for name, model1 in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model1, X_train, y_train, cv=kfold, 
                              scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s accuracy with KFold: %f (%f)\n' % (name, (cv_results.mean()*100), 
                                              cv_results.std()))



##############################Confusion Matrix#################################
from sklearn.metrics import confusion_matrix

#Menghitung Confusion Matrix
svm_confusion_matrix = confusion_matrix(y_test, predictions)

#Hitung nilai Accuracy
from sklearn.metrics import accuracy_score
Accuracy_confusion = accuracy_score(y_test, predictions)
print("Accuracy of Confusion: %.2f%%" % (Accuracy_confusion*100))

#Hitung nilai Precision
from sklearn.metrics import precision_score
Precision_confusion = precision_score(y_test, predictions, pos_label=1, 
                                      average='macro')
print("Precision of Confusion: %.2f%%" % (Precision_confusion*100))

#Hitung nilai Recall
from sklearn.metrics import recall_score
Recall_confusion = recall_score(y_test,predictions, pos_label=1, 
                                average='macro')
print("Recall of Confusion: %.2f%%\n" % (Recall_confusion*100))

#Plot Confusion Matrix
import matplotlib.pyplot as plt 
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(svm_confusion_matrix, annot=True, fmt="1.0f", cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()



##############################ROC##############################################
#Ubah tulisan jadi angka vektor
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
y_vector = cv.fit_transform(y).toarray()

#Memberikan Label
from sklearn.preprocessing import label_binarize
y_binarizer = label_binarize(y_vector, classes=[0, 1, 2, 3, 4])
n_classes = y_vector.shape[1]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y_binarizer, 
                                                        test_size=0.20, 
                                                        random_state=1)

#Menghitung nilai ROC
from sklearn.metrics import roc_auc_score
result_predict = model.predict_proba(X_test)
roc_score = roc_auc_score(y_test, result_predict,multi_class='ovo', 
                          average="macro")
result_predict.shape
print("ROC Score from SVM = %f%%" % (roc_score*100))

#Plot Grafik ROC
from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], result_predict[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["nano"], tpr["nano"], _ = roc_curve(y_test1.ravel(), result_predict.ravel())
roc_auc["nano"] = auc(fpr["nano"], tpr["nano"])

import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='Predicted (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, label='Predicted', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Prediction')
plt.ylabel('True Data')
plt.title('ROC')
plt.show()





###############################################################################
##############################USER INTERFACE###################################


##############################Input level dan rekomendasi ke dataframe#########
import numpy as np
vector = np.vectorize(np.int)
y_olah = vector(predictions_plot)
pre_prediksi = np.array(y_olah).astype('str')
pre_prediksi = pd.DataFrame(pre_prediksi,columns=['Berat','Berbahaya','Menengah',
                                                  'Normal','Ringan'])
pre_prediksi["level"] = pre_prediksi["Berat"] + pre_prediksi["Berbahaya"] + pre_prediksi["Menengah"] + pre_prediksi["Normal"] + pre_prediksi["Ringan"]
pre_prediksi['level'] = pre_prediksi['level'].replace(
    ['10000','01000','00100','00010','00001'],
    ['BERAT','BERBAHAYA','MENENGAH','NORMAL','RINGAN'])
pre_prediksi["perawatan"] = pre_prediksi["Berat"] + pre_prediksi["Berbahaya"] + pre_prediksi["Menengah"] + pre_prediksi["Normal"] + pre_prediksi["Ringan"]
pre_prediksi['perawatan'] = pre_prediksi['perawatan'].replace(
    ['10000','01000','00100','00010','00001'],
    ['Cek conveyor, ekstra hati-hati',
     'Hentikan penggunaan conveyor, Cek conveyor keseluruhan',
     'Mulai cek conveyor, mulai ekstra hati-hati',
     'Selalu waspada',
     'Mulai waspada, lapor ke teknisi'])
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