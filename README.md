# Laporan Akhir Machine Learning :  Anime Recommendation System
## Proyek Overview
![anime](https://github.com/user-attachments/assets/ba253f9e-5471-47ea-b4d8-bafa2bbca127)


# Domain Proyek
Anime merupakan salah satu konten yang paling banyak dikonsumsi. Anime baru terus bertambah setiap tahun membuat pengguna kesulitan 
menemukan tontonan yang sesuai dengan preferensi mereka, dan hal ini berdampak langsung pada keberhasilan bisnis platform penyedia 
konten. Dengan menghadirkan sistem rekomendasi yang efektif dan personal, platform dapat meningkatkan retensi pengguna dan durasi 
waktu menonton, yang pada akhirnya mendorong pertumbuhan pendapatan. Netflix menunjukkan bahwa lebih dari 80% tontonan 
berasal dari sistem rekomendasi, menjadikannya komponen vital dalam strategi bisnis digital mereka [1]. Selain itu, sistem ini juga 
membuka peluang monetisasi melalui iklan tertarget dan promosi konten eksklusif berbasis minat pengguna. MyAnimeList telah mencatat 
lebih dari 10.000 judul anime dalam databasenya [2], maka kehadiran sistem penyaring cerdas sangat diperlukan untuk menciptakan 
pengalaman pengguna yang efisien dan memuaskan [3]. Sistem rekomendasi yang baik dapat menjadi alat bisnis 
strategis yang tidak hanya meningkatkan kepuasan pengguna, tetapi juga memperkuat loyalitas dan nilai ekonomi jangka panjang.

# Mengapa masalah ini harus diselesaikan
 1. Pendorong pertumbuhan pendapatan bisnis
 2. Penciptaan pengalaman pengguna yang efisien dan memuaskan

## Business Understanding
### Problem Statements
Rumusan masalah dari masalah latar belakang diatas adalah
  1. bagaimana pesebaran anime berdasarkan genre dan type distribution
  2. bagaimana member komunitas mempengaruhi rating
  3. bagaimana cara membuat sistem rekomendasi terbaik yang dapat diimplesikan ? 

### Goals
Berdasarkan problem statements, berikut tujuan dibuatnya proyek ini.
  1. Mengetahui persebaran anime berdasarkan genre dan type distribution
  2. Mengetahui member komunitas anime
  3. Menggunakan algoritma cosine similiarity maupun pemodelan machine learning untuk membuat sistem rekomendasi

### Solution Approach

1. Mengimplementasikan Exploratory Data Analysis (EDA) untuk analisis dan visualisasi data.
2. Mengimplementasikan content-based filtering approach menggunakan algoritma cosine similarity.
3. Mengimplementasikan collaborative-based filtering approach menggunakan algoritma deep learning.
4. Evaluasi Performa Model setelah model dibangun, evaluasi performa akan dilakukan menggunakan metrik seperti Precision dan
   Root Mean Squared Error. Ini akan memberikan wawasan tentang efektivitas model dalam merekomendasikan anime yang relevan
   kepada pengguna.
   
### Metrik
Metrik evaluasi yang digunakan adalah precision untuk content based filtering dan RMSE untuk collaborative filtering

## Data Understanding
Dataset yang digunakan untuk membuat sistem rekomendasi anime pada responden diambil dari platform kaggle yang dipublikasikan oleh Cooperunion [berikut](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database).

### Keterangan Variabel
Dataset ini memiliki 2 file yaitu rating.csv dan anime.csv

Keterangan rating
Variabel | Keterangan
----------|----------
user_id | nomor unik masing-masing user
anime_id | nomor unik masing-masing anime di MyAnimeList
rating | (-1 menandakan tidak rating walaupun menonton) rating user dari 10

| No | Column              | Dtype    |
|----|---------------------|----------|
| 0  | user_id             | int64    |
| 1  | anime_id            | int64    | 
| 2  | rating              | int64    | 

Jumlah baris: 7813737
Jumlah kolom: 3

Semua variabel bertipe int64 dan baris pada data berjumlah 7813737 denngan 3 kolom

Keterangan anime
Variabel | Keterangan
----------|----------
anime_id | nomor unik masing-masing anime di MyAnimeList
name | nama lengkap anime
genre | daftar genre yang dipisahkan oleh koma
type | movie, TV, OVA, etc
episodes | berapa jumlah episode
rating | rata-rata rating dari 10 
members | jumlah member komunitas anime baris ini
<br>

| No | Column             | Non-Null Count | Dtype    |
|----|--------------------|----------------|----------|
| 0  | anime_id           |12294  non-null  | int64     |
| 1  | name               |12294  non-null  | object   | 
| 2  | genre              |12232   non-null  | object   | 
| 3  | type               |12269   non-null  | object   | 
| 4  | episodes           |12294   non-null  | object  |
| 5  | rating             |12064   non-null  | float64  |
| 6  | members            |12294  non-null  | int64    |

Jumlah baris: 12294
Jumlah kolom: 7

terdapat 4 tipe data objek yaitu name, genre, type dan episodes. 2 tipe data int64 yaitu anime_id dan members dan 1 tipe data float64 
yaitu rating. 
### Tabel statistik
Tabel statistik
- Count adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom.
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval - dalam empat bagian sebaran yang sama.
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.

### Statistik Data rating
Selanjutnya akan ditampilkan statistik data numerikal secara umum:
| Statistic |	user_id      | anime_id         |rating|
|-----------|-------------|-----------|-----------------|
|count| 7813737.00 |7813737.00 7813737.00|
|mean  |  36727.96  |  8909.07    |   6.14|
|std    | 20997.95   | 8883.95     |  3.73|
|min     |    1.00    |   1.00      |-1.00|
|25%     |18974.00    |1240.00      | 6.00|
|50%     |36791.00    |6213.00      | 7.00|
|75%     |54757.00   |14093.00      | 9.00|
|max     |73516.00   |34519.00      |10.00|

Dari informasi diatas dapat disimpulkan bahwa data 
- pada kolom *rating* menunjukkan rentang rating dari -1 hingga 10 dengan rata-rata 6.14 dengan -1 menandakan user belum memberikan rating

### Statistik Data anime
Selanjutnya akan ditampilkan statistik data numerikal secara umum:
| anime_id    |	rating       | members    |
|-----------|-------------|-----------|
|count  |12294.00| 12064.00   |12294.00|
|mean   |14058.22|     6.47   |18071.34|
|std    |11455.29|     1.03   |54820.68|
|min    |    1.00|     1.67     |  5.00|
|25%    | 3484.25|     5.88    | 225.00|
|50%    |10260.50|     6.57   | 1550.00|
|75%    |24794.50|     7.18  |  9437.00|
|max    |34527.00|    10.00 |1013917.00|

Dapat dilihat dari informasi diatas bahwa 
- pada kolom *member* menunjukka komunitas terbanyak ada di angka 1.013.917 yaitu 1 juta lebih member dengan yang paling sedikit 5 member dan rata-rata sebanyak 14058 member. 
- pada kolom *rating* menunjukka menunjukka rata-rata rating berada di 6.47 dengan rating paling kecil ada di 1.67 dan rating paling tinggi ada di 10

#### memeriksa data duplikasi
![duplikasi](https://github.com/user-attachments/assets/2ff3c2f8-0100-4a32-ac2b-d41aecb0a03f)

dari hasil diatas disimpulkan bahwa data tidak terjadi duplikasi maka akan dilanjutkan pengecekan missing value pada data.

#### memeriksa missing value
![missing value](https://github.com/user-attachments/assets/4cd5ef0d-ee12-4121-a771-b297371b7918)

Terdapat beberapa missing value yaitu
 - genre : 62 
 - type : 25 
 - rating : 230


## Exploratory Data Analysis (EDA)
### Distribusi rating anime 
![ratin anime](https://github.com/user-attachments/assets/499b539f-646e-4100-9f1c-a6054640d24c)

<br>
Distribusi left-skewed dengan kebanyakan di rentang 6-7

### Distribusi member komunitas anime

![komunitas](https://github.com/user-attachments/assets/920a856e-ad52-43e8-a187-7b601f42f097)

<br>
Anime dengan member komunitas terbanyak jatuh kepada 'Death Note' dengan jumlah member 1.013.917

### Distribusi anime berdasarkan genre
Karena genre berupa list dengan seperator ',' maka dilakukan explode atau unnesting

![code](https://github.com/user-attachments/assets/32d13207-f532-47d5-820e-37cd0181e383)

![genre](https://github.com/user-attachments/assets/660423a5-76fb-4ea6-8827-415c59f308ae)

<br>
Berdasarkan gambar diatas genre comedy memiliki anime terbanyak dengan jumlah 32.7%

### Distribusi anime berdasarkan type

![type](https://github.com/user-attachments/assets/f9cdc37e-537e-431a-8b2b-4bae58dae94b)

<br>
Berdasarkan gambar diatas tipe anime TV memiliki distribusi terbanyak dengan jumlah 31.1%

### Top anime berdasarkan rating

![image](https://github.com/user-attachments/assets/482c835e-f084-4f33-abf7-1e7c6bd399c9)

<br>
Anime terbaik berdasarkan rating adalah Taka no Tsume8: Yoshida-kun no X-Files dengan rating 10 diikuti dengan spoon-hime no swing kitchen pada urutan kedua


### user rating

![image](https://github.com/user-attachments/assets/2b13b8ba-f7bd-47e4-ada4-ebf84b37fb13)

<br>
Distribusi left-skewed dengan kebanyakan di rentang 7-8 dan terdapat outliers yaitu -1 yang nantinya akan di hilangkan


## Data Preparation
Karena berbeda antara content-based filtering dengan collaborative filtering, maka data preparation dari kedua approach tersebut akan dilakukan secara masing-masing. Teknik Data preparation yang dilakukan terdiri dari:
- Menghapus baris yang memiliki genre null,
- Imputasi pada type yang kosong dengan type paling populer
  
  ![image](https://github.com/user-attachments/assets/5afa0468-1bc8-492b-b81d-155a9451134d)

- Drop duplikat
- Drop null pada rating
  
  ![image](https://github.com/user-attachments/assets/0051740b-b387-4667-bd5d-00c0bdd7d874)
  
- Menggunakan regex untuk judul yang memiliki simbol
  
![image](https://github.com/user-attachments/assets/3d941fdb-5255-49de-b8ba-9f3c09f40644)

- TF-IDF Vectorizer 
- Encoding Data User Rating
- Train-test-split Data User Rating

### 1. Content-Based Filtering
Untuk content-based filtering, kita akan fokus pada genre yang diproses dengan memisahkan genre-genre berdasarkan koma, kemudian untuk setiap genre menghilangkan spasi di awal/akhir dan mengganti spasi di tengah dengan garis bawah, lalu menggabungkan kembali semua genre tersebut menjadi satu string 
dengan spasi tunggal sebagai pemisah. Contoh "Shounen Ai, Adventure" menjadi "Shounen_Ai Adventure" untuk menjadi dasar pembuatan sistem rekomendasi tersebut. 

Selanjutnya, digunakan TfidfVectorizer() pada kolom genre untuk menghasilkan output berupa angka antara 0 - 1. Lalu, dibentuk dataframe yang berisi kolom genre yang telah dilakukan vektorisasi dengan TfidfVectorizer() sebagai kolom dan seluruh nama anime 
sebagai barisnya. Hal ini dilakukan karena akan digunakan cosine similarity pada content-based filtering, dimana cosine similarity memerlukan bentuk angka agar dapat dihitung. Contoh dari dataframe dapat dilihat pada tabel berikut.
![tabel](https://github.com/user-attachments/assets/fe57e384-b89e-46e3-9cb1-b073a875b420)

**Mengapa diperlukan Mengubah data kedalam representasi numerik?**

- Data perlu diubah kedalam representasi numerik karena sistem rekomendasi berbasis konten membutuhkan representasi numerik dari teks atau fitur kategori agar dapat mengukur kemiripan antar-item. Misalnya, dalam rekomendasi anime, kategori seperti "Adventure," "Action," atau "Supernatural" diubah menjadi nilai numerik untuk dihitung kemiripannya.

### 2. Collaborative Filtering
Untuk collaborative filtering, kita akan fokus pada user_id, dan rating .


Karena **user_id** dan **anime_id** memiliki tipe data string dan unik, maka dilakukan encoding terhadap kedua kolom tersebut, kemudian dibentuk dataframe yang berisi kolom **user_id** yang sudah diencoding, kolom **anime_id** yang sudah diencoding, dan **Rating**. Contoh dari dataframe dapat dilihat pada tabel berikut.

![image](https://github.com/user-attachments/assets/0f1eef58-a005-4335-ba41-a280436dd7e7)


**Mengapa diperlukan melakukan Encoding data?**
- Encoding data perlu dilakukan karena pada Collaborative Filtering, model harus belajar dari pola interaksi pengguna terhadap item. Data perlu diubah ke dalam bentuk numerik agar model neural network dapat memprosesnya.

-Data yang telah di-encode selanjutnya dipisahkan menjadi data latih sebesar 80% untuk proses pembelajaran model, dan data uji sebesar 20% untuk mengevaluasi seberapa baik model yang dibuat mampu bekerja pada data yang belum pernah digunakan dalam pelatihan.

**Mengapa diperlukan melakukan Train-test-split data?**

- MUntuk mendapatkan gambaran yang lebih tepat tentang seberapa baik model yang kita buat, data dibagi menjadi set pelatihan dan pengujian, di mana set pengujian berisi data baru untuk menilai kinerja model secara akurat


## Modelling and Result

### 1. Content-Based Filtering

Content-based filtering menggunakan cosine similarity sebagai algoritma untuk membuat sistem rekomendasi berdasarkan content-based filtering. Cosine similarity mengukur kesamaan antara dua vektor dan menentukan jika kedua vektor tersebut menunjuk ke arah yang sama. 
Ia menghitung sudut cosinus antara dua vektor. Semakin kecil sudut cosinus, semakin besar nilai cosine similarity. Cosine similarity dirumuskan sebagai berikut.

$$Cos (\theta) = \frac{\sum_1^n a_ib_i}{\sqrt{\sum_1^n a_i^2}\sqrt{\sum_1^n b_i^2}}$$

Kita akan memakai cosine_similarity untuk menghitung kemiripan antar vektor dalam matriks. Metode ini unggul karena menghasilkan output ternormalisasi (-1 hingga 1) yang mudah ditafsirkan, mudah digunakan, dan efisien untuk data sparse berdimensi tinggi seperti TF-IDF. 
Namun, kekurangannya adalah semua faktor dianggap sama penting,sensitif pada perubahan 'sudut vektor', dan kurang cocok untuk data negatif. Setelah sistem rekomendasi terbentuk, sistem ini akan diuji untuk menampilkan 10 rekomendasi teratas berdasarkan genre. Berikut hasilnya

**recommendations_content('Naruto')**

![content_based](https://github.com/user-attachments/assets/55a805be-b90c-4e69-8962-c063b623406f)


### 2. Collaborative Filtering

Collaborative Filtering menggunakan deep learning, tepatnya embedding layer untuk membuat model deep learning. Embedding layer merupakan tipe layer pada deep learning yang digunakan untuk mentransformasikan data kategorikal menjadi vektor dengan nilai kontinu. 
Pada python, kita menggunakan **tensorflow.keras.layers Embedding** untuk membentuk embedding layer. Embedding Layer memiliki kelebihan seperti  dapat digunakan di berbagai macam algoritma deep learning,mengurangi kompleksitas model, dan menangkap hubungan semantic pada data.
Namun, embedding layer juga memiliki beberapa kelemahan, seperti membutuhkan data yang banyak, sensitif terhadap hyperparameter, dan cold start problem. Setelah model dibentuk dan dilatih, diperoleh hasil **root_mean_squared_error: 0.1254** untuk data training dan 
**val_root_mean_squared_error:  0.1398** untuk data testing. Nilai tersebut sudah bagus untuk digunakan dalam sistem rekomendasi, sehingga dapat dibentuk sistem rekomendasi berdasarkan model tersebut. Selanjutnya, akan diuji sistem rekomendasi ini untuk menampilkan 
top 10 rekomendasi anime berdasarkan user lain. Diperoleh hasil berikut.

![collaborative_filtering](https://github.com/user-attachments/assets/7eb34a3d-cb6f-4575-91c3-2425562f8f0f)


## Evaluation

### 1. Content-Based Filtering

Pada content-based filtering, model ini hanya menggunakan metrik Precision untuk mengetahui seberapa baik perforam model tersebut. Presisi adalah metrik yang biasa digunakan untuk mengevaluasi kinerja model pengelompokan. 
Metrik ini menghitung rasio antara nilai ground truth (nilai sebenarnya) dengan nilai prediksi yang positf. Perhitungan rasio ini dijabarkan melalui rumus di bawah ini:

$$ Precision = \frac{TP}{TP + FP} $$

Dimana:

- TP (*True Positive*), jumlah kejadian positif yang diprediksi dengan benar.
- FP (*False Positive*), jumlah kejadian positif yang diprediksi dengan salah.

Berdasarkan hasil yang terdapat pada tahap Model and Result dapat dilihat bahwasanya besar presisi jika dihitung adalah 10/10 untuk rekomendasi Top-10. Ini menunjukan sistem mampu memberikan rekomendasi sesuai dengan genre.


### 2. Collaborative Filtering

Pada collaborative filtering, metrik evaluasi yang digunakan adalah Root Mean Squared Error (RMSE). 

#### Sekilas tentang RMSE

Root Mean Squared Error (RMSE) merupakan salah satu metode untuk menghitung error pada pelatihan model dengan cara menghitung jarak rata-rata antara nilai yang diprediksi dengan nilai sesungguhnya. RMSE dirumuskan sebagai berikut.

$$RMSE = \sqrt{\frac{\sum_{i=1}^n{(y_i - \hat{y_i})}^2}{N}}$$

Keterangan:
* $y_i$: Nilai sesungguhnya pada observasi ke-i
* $\hat{y_i}$: Nilai prediksi pada observasi ke-i
* $N$: Jumlah observasi

Jika nilai prediksi sangat mendekati nilai sesungguhnya, maka nilai dari $(y_i - \hat{y_i})$ akan semakin mengecil. Artinya, semakin kecil nilai dari RSME atau bahkan mendekati nol, maka model yang digunakan telah akurat dan baik.

#### Penerapan Evaluasi Model dengan RMSE

Pada collaborative filtering, setelah melatih model sebanyak 10 epoch, diperoleh hasil **RMSE = 0.1254 ** untuk data training dan **RMSE = 0.1398** untuk data testing. Jika dilihat menggunakan grafik.

![grafik rmse](https://github.com/user-attachments/assets/f36634ac-ec4d-40aa-9fa4-1486bfe4710e)


Dari gambar tersebut, terlihat bahwa nilai RMSE pada data training selalu menurun, sementara nilai RSME pada data testing awalnya menurun, namun setelah 4 epoch, nilai RSME mulai stagnan. Meski RSME pada data testing lebih besar dari data training, 
namun karena mendekati 0, maka model yang digunakan telah baik dan akurat untuk membuat sistem rekomendasi.


## Kesimpulan
1. Perilaku konsumsi penggemar anime tampaknya berada pada genre comedy (32.7%) dan tipe anime yang ditonton adalah TV (30.9%)
2. Anime yang paling banyak member komunitasnya adalah 'Death Note' dengan jumlah member 1.013.917, produser anime bisa membuat anime baru dengan genre atau tema yang sama untuk memanfaatkan jumlah member yang besar demi keuntungan
3. Sistem rekomendasi dapat diimplementasikan dengan menggunakan 2 approach, yaitu content-based filtering approach menggunakan cosine similarity dan collaborativer filtering approach menggunakan embedding layer untuk memberikan sistem rekomendasi terbaik.

## Referensi

[1] Gufy. How Netflix’s Recommendation Engine Drives Success. Diakses pada 1 Juni 2025 dari https://www.gufy.com.au/post/netflixs-recommendation-engine#:~:text=In%20the%20world%20of%20streaming,also%20significantly%20improved%20user%20retention.

[2] MyAnimeList Stats. (2024). Diakses pada 1 Juni 2025 dari https://myanimelist.net/topanime.php?limit=20000

[3] Ricci, F., Rokach, L., & Shapira, B. (2011). Recommender Systems Handbook. Springer.

[4] Dicoding. Diakses pada 1 Juni 2025 dari https://www.dicoding.com/academies/319/corridor

