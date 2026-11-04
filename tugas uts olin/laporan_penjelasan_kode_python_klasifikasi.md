# LAPORAN PENJELASAN LANGKAH-LANGKAH KODE MACHINE LEARNING
**Topik:** Klasifikasi Kelulusan Mahasiswa (*Pass/Fail Prediction*)

---

## BAB 1: IMPORT LIBRARY DAN PEMUATAN DATA

### Langkah 1: Mengimpor Library Utama (In [1] & In [6])
* **Kode:**
  ```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  import warnings
  warnings.filterwarnings('ignore')
  ```
* **Penjelasan:**
  * `pandas` ($pd$): Digunakan untuk pengolahan data berbentuk tabel (DataFrame).
  * `numpy` ($np$): Digunakan untuk komputasi numerik dan manipulasi array/nilai $NaN$.
  * `matplotlib.pyplot` ($plt$) & `seaborn` ($sns$): Digunakan untuk membuat visualisasi grafik dan plot statistik.
  * `warnings.filterwarnings('ignore')`: Berfungsi untuk menyembunyikan pesan peringatan (*warning*) agar tampilan output bersih.

---

### Langkah 2: Mengimpor Library Machine Learning (In [2], In [4], In [5])
* **Kode:**
  ```python
  from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
  from sklearn.preprocessing import StandardScaler
  from sklearn.impute import SimpleImputer
  from sklearn.pipeline import Pipeline
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
  from sklearn.inspection import permutation_importance
  ```
* **Penjelasan:**
  * **Model & Pipeline:** Mengimpor Algoritma *Logistic Regression* dan *Random Forest Classifier*, serta `Pipeline` untuk menggabungkan langkah-langkah pemrosesan data.
  * **Preprocessing:** `SimpleImputer` untuk mengisi data yang hilang (*missing values*) dan `StandardScaler` untuk menyamakan skala fitur (*feature scaling*).
  * **Evaluasi & Tuning:** `train_test_split` untuk membagi data, `GridSearchCV` & `StratifiedKFold` untuk *hyperparameter tuning* dan validasi silang, serta metrik evaluasi ($F1-score$, $Accuracy$, $Precision$, $Recall$).

---

### Langkah 3: Membaca Dataset dan Pemeriksaan Awal (In [7] - In [12])
* **Kode:**
  ```python
  df = pd.read_csv('original.csv')
  print("Ukuran Dataset:", df.shape)
  display(df.head())
  df.info()
  print(df.isnull().sum())
  print(df['label (fail=1, pass=0)'].value_counts())
  ```
* **Penjelasan:**
  * Data dibaca dari file `original.csv`. Ukuran awal dataset adalah **87 baris dan 22 kolom**.
  * `df.head()` menampilkan 5 baris pertama data.
  * `df.info()` dan `df.isnull().sum()` memeriksa tipe data dan jumlah data kosong (*missing values*). Ditemukan beberapa kolom memiliki baris kosong, dan ada 2 kolom yang sepenuhnya kosong (`Unnamed: 20` dan `Unnamed: 21`).
  * `value_counts()` menampilkan distribusi target: **68 mahasiswa Lulus (0)** dan **12 mahasiswa Gagal (1)**.

---

## BAB 2: EXPLORATORY DATA ANALYSIS (EDA) DAN PEMBERSIHAN DATA

### Langkah 4: Visualisasi Distribusi Target (In [13])
* **Kode:**
  ```python
  df['label (fail=1, pass=0)'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
  plt.title('Distribusi Kelulusan Mahasiswa')
  plt.xlabel('Label (0 = Lulus, 1 = Gagal)')
  plt.ylabel('Jumlah Mahasiswa')
  plt.show()
  ```
* **Penjelasan:**
  Membuat diagram batang (*bar chart*) untuk melihat keseimbangan label. Terlihat bahwa dataset mengalami **Imbalanced Data** (jumlah kelas $0$ jauh lebih banyak dibanding kelas $1$).

---

### Langkah 5: Pembersihan Data / Data Cleaning (In [14] - In [21])
* **Kode:**
  ```python
  df = df.dropna(how='all')
  df = df.dropna(axis=1, how='all')
  df.columns = df.columns.str.strip()
  df = df.dropna(subset=['label (fail=1, pass=0)'])
  df = df.drop_duplicates()

  kolom_kuintil = [kolom for kolom in df.columns if 'Quintile' in kolom]
  for kolom in kolom_kuintil:
      df[kolom] = df[kolom].replace(0, np.nan)
  ```
* **Penjelasan:**
  * Menghapus baris dan kolom yang kosong secara keseluruhan.
  * `str.strip()` membersihkan spasi berlebih pada nama-nama kolom.
  * Menghapus baris yang tidak memiliki label target.
  * Menghapus baris duplikat. Setelah dibersihkan, ukuran data menjadi **80 baris dan 20 kolom**.
  * Mengganti nilai $0$ pada kolom berjenis *Quintile* menjadi $NaN$ karena nilai $0$ diindikasikan sebagai data hilang.

---

### Langkah 6: Visualisasi Distribusi Fitur dan Korelasi (In [22] - In [26])
* **Kode:**
  ```python
  # Histogram Usia, Login, Jam Belajar
  sns.histplot(df['Age'], bins=10, kde=True)
  sns.histplot(df['Count of Module Area Logins'], bins=10, kde=True)
  sns.histplot(df['Total Hours in Module Area'], bins=10, kde=True)

  # Boxplot Kehadiran vs Kelulusan
  sns.boxplot(x='label (fail=1, pass=0)', y='# of presence', data=df)

  # Heatmap Korelasi
  plt.figure(figsize=(12,8))
  korelasi = df.select_dtypes(include='number').corr()
  sns.heatmap(korelasi, annot=False, cmap='coolwarm')
  ```
* **Penjelasan:**
  * Visualisasi histogram digunakan untuk melihat sebaran data numerik seperti Usia, Jumlah Login, dan Total Jam Belajar.
  * *Boxplot* digunakan untuk membandingkan tingkat kehadiran antara mahasiswa yang lulus vs gagal.
  * *Heatmap* digunakan untuk melihat hubungan (korelasi) antar fitur numerik.

---

## BAB 3: PERSIAPAN DATA (DATA PREPARATION)

### Langkah 7: Seleksi Fitur dan Pembagian Target (In [27] - In [28])
* **Kode:**
  ```python
  df = df.drop(columns=['TUNDRA LSOA Quintile', '# of Absence', 'Percent Attended'])
  X = df.drop(columns=['label (fail=1, pass=0)'])
  y = df['label (fail=1, pass=0)']
  ```
* **Penjelasan:**
  * Menghapus fitur yang redundan atau memiliki multikolinearitas tinggi (seperti `# of Absence` dan `Percent Attended` yang berkorelasi langsung dengan `# of presence`).
  * Memisahkan variabel independen / fitur ($X$, sebanyak 16 fitur) dan variabel dependen / target ($y$).

---

### Langkah 8: Pembagian Data Latih dan Data Uji (In [29])
* **Kode:**
  ```python
  X_train, X_test, y_train, y_test = train_test_split(
      X, y,
      test_size=0.2,
      random_state=42,
      stratify=y
  )
  ```
* **Penjelasan:**
  * Data dibagi menjadi **80% Data Latih (64 sampel)** dan **20% Data Uji (16 sampel)**.
  * `stratify=y` digunakan agar proporsi kelas target ($0$ dan $1$) pada data latih dan data uji tetap seimbang selaras dengan dataset asli.

---

## BAB 4: PEMODELAN (MODELING)

### Langkah 9: Pemodelan dengan Logistic Regression (In [30] - In [32])
* **Kode:**
  ```python
  model_lr = Pipeline([
      ('imputer', SimpleImputer(strategy='median')),
      ('scaler', StandardScaler()),
      ('model', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
  ])

  model_lr.fit(X_train, y_train)
  pred_lr = model_lr.predict(X_test)
  ```
* **Penjelasan:**
  * Menggunakan `Pipeline` yang terdiri dari:
    1. `SimpleImputer`: Mengisi nilai kosong dengan nilai median.
    2. `StandardScaler`: Normalisasi skala data.
    3. `LogisticRegression`: Pelatihan model dengan parameter `class_weight='balanced'` untuk menangani masalah ketidakseimbangan kelas (*imbalanced data*).
  * Model dilatih menggunakan `fit()` dan melakukan prediksi pada data uji dengan `predict()`.

---

### Langkah 10: Pemodelan & Tuning Random Forest dengan GridSearchCV (In [33] - In [38])
* **Kode:**
  ```python
  # Definisi Pipeline Random Forest
  model_rf = Pipeline([
      ('imputer', SimpleImputer(strategy='median')),
      ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
  ])

  # Parameter Grid
  parameter = {
      'model__n_estimators': [100, 200],
      'model__max_depth': [None, 3, 5],
      'model__min_samples_leaf': [1, 2]
  }

  # Cross Validation & Grid Search
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  grid = GridSearchCV(model_rf, parameter, scoring='f1', cv=cv, n_jobs=-1)

  # Pelatihan
  grid.fit(X_train, y_train)
  best_rf = grid.best_estimator_
  ```
* **Penjelasan:**
  * Membuat pipeline algoritma **Random Forest**.
  * Menentukan variasi parameter yang akan diuji (`n_estimators`, `max_depth`, `min_samples_leaf`).
  * Menggunakan **5-Fold Stratified Cross Validation** untuk memastikan evaluasi objektif.
  * `GridSearchCV` mencari kombinasi parameter terbaik berdasarkan metrik `f1-score`.
  * Hasil tuning menghasilkan parameter terbaik: `{'model__max_depth': None, 'model__min_samples_leaf': 1, 'model__n_estimators': 100}`.