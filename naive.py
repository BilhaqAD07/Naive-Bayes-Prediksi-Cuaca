import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score

print("================================================")
print("PREDIKSI HUJAN MENGGUNAKAN ALGORITMA NAIVE BAYES")
print("================================================")

data = pd.read_clipboard(sep='\t')  # jika data tersimpan pada clipboard

# Clean data
data['suhu'] = data['suhu'].str.replace(',', '.').astype(float)
data['penyinaran'] = data['penyinaran'].str.replace(',', '.').astype(float)

samplesize = round(0.8 * data.shape[0])
np.random.seed(12345)
index = np.random.choice(
    np.arange(data.shape[0]), size=samplesize, replace=False)
training = data.iloc[index, :]

# --------------------- Model --------------------
testing = data.iloc[np.setdiff1d(np.arange(data.shape[0]), index), :]
naive = MultinomialNB()
naive.fit(training.drop('status', axis=1), training['status'])

# --------------------- Data Training --------------------
xx = training.drop('status', axis=1)
pred22 = naive.predict(xx)
cm22 = confusion_matrix(training['status'], pred22)
accuracy22 = np.sum(np.diag(cm22)) / np.sum(cm22)
precision22 = precision_score(training['status'], pred22, average='macro', zero_division=0)
recall22 = recall_score(training['status'], pred22, average='macro', zero_division=0)
print({
    'Matrix_Data Train': cm22,
    'Akurasi': accuracy22*100,
    'Presisi': precision22,
    'Recall': recall22,
    'Total_data': np.sum(cm22)
})

# --------------------- Data Testing --------------------
xtest = testing.drop('status', axis=1)
predtest = naive.predict(xtest)
cmtest = confusion_matrix(testing['status'], predtest)
accuracytest = np.sum(np.diag(cmtest)) / np.sum(cmtest)
precisiontest = precision_score(testing['status'], predtest, average='macro', zero_division=0)
recalltest = recall_score(testing['status'], predtest, average='macro', zero_division=0)
print({
    'Matrix_Data Testing': cmtest,
    'Akurasi': accuracytest*100,
    'Presisi': precisiontest,
    'Recall': recalltest,
    'Total_data': np.sum(cmtest)
})

# --------------------- Prediksi berdasarkan input hari ini --------------------
print('\n')
input_suhu = float(input("Masukkan suhu saat ini: "))
input_kelembaban = float(input("Masukkan kelembaban saat ini: "))
input_penyinaran = float(input("Masukkan nilai penyinaran saat ini: "))
input_kec_angin = int(input("Masukkan kecepatan angin saat ini: "))

input_data = pd.DataFrame({'suhu': [input_suhu],
                           'kelembaban': [input_kelembaban],
                           'penyinaran': [input_penyinaran],
                           'kec_angin': [input_kec_angin]}, 
                          columns=['suhu', 'kelembaban', 'penyinaran', 'kec_angin'])
prediksi = naive.predict(input_data)

print("\nHasil prediksi cuaca berdasarkan input hari ini:")
print(prediksi)
