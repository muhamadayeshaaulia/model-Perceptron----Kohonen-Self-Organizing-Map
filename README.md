## Aplikasi Deteksi Produktivitas dan Burnout Mahasiswa Menggunakan Perceptron dan Kohonen Self-Organizing Map

## 1 . Virtual environment

    ```python
    python3 -m venv venv
    source venv/bin/activate
    ```
## 2 . Install Library Python

    ```python
    pip install numpy pandas matplotlib streamlit
    ```
## 3 . Membuat beberapa kebutuhan file
    ```python
    touch app.py dataset.csv perceptron.py som.py helper.py README.md
    ```
## 4 . Membuat Dataset.csv
    ```python
    //membuat dataset dengan beberapa contoh data
    jam_tidur,mood,stres,jam_belajar,jam_hp,jumlah_tugas,label
    8,9,2,5,2,1,1
    7,8,3,4,3,2,1
    7,7,4,4,3,3,1
    6,7,4,3,4,2,1
    8,8,3,5,2,2,1
    6,6,5,3,4,3,1
    7,8,2,4,2,1,1
    6,7,3,4,3,2,1
    5,6,6,3,5,4,0
    5,5,7,2,6,5,0
    4,5,8,2,7,5,0
    4,4,9,1,8,6,0
    3,3,10,1,9,7,0
    5,4,8,2,8,6,0
    6,5,7,2,6,5,0
    4,5,9,1,7,6,0
    8,9,2,6,2,1,1
    7,8,3,5,3,2,1
    6,6,6,3,5,4,0
    5,5,8,2,7,5,0
    7,7,4,4,3,2,1
    6,8,3,5,2,3,1
    4,4,8,1,8,6,0
    3,4,9,1,9,7,0
    8,8,2,5,2,1,1
    7,7,3,4,3,2,1
    6,5,7,2,6,5,0
    5,6,6,2,5,4,0
    8,9,1,6,1,1,1
    4,3,10,1,9,8,0

    ```