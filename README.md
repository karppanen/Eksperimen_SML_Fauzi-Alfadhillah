# Eksperimen SML - Fauzi Alfadhillah

Repository ini berisi eksperimen dan otomatisasi preprocessing untuk submission kelas **Membangun Sistem Machine Learning** Dicoding.

## Struktur Repository

```text
Eksperimen_SML_Fauzi-Alfadhillah
├── .github/workflows/preprocessing.yml
├── CustomerSegmentation_raw.csv
└── preprocessing
    ├── Eksperimen_Fauzi-Alfadhillah.ipynb
    ├── automate_Fauzi-Alfadhillah.py
    ├── requirements.txt
    └── CustomerSegmentation_preprocessing
        ├── customer_segmentation_preprocessed.csv
        └── label_mapping.csv
```

## Dataset

Dataset yang digunakan adalah data segmentasi pelanggan dengan fitur:

- `Age`
- `Total Salary (IDR)`
- `Total Spending (IDR)`
- `Frequency (Yearly)`
- `Membership Status`

Target prediksi adalah `Membership Status`.

## Tahapan Eksperimen

Notebook eksperimen berisi:

1. Pengenalan dataset
2. Import library
3. Data loading
4. Exploratory Data Analysis
5. Data preprocessing

Preprocessing yang dilakukan:

- Menghapus data duplikat
- Mengecek dan menangani missing value
- Memastikan fitur numerik bertipe angka
- Encoding target `Membership Status`
- Standardisasi fitur numerik
- Menyimpan dataset hasil preprocessing

## Menjalankan Preprocessing

Install dependency:

```bash
pip install -r preprocessing/requirements.txt
```

Jalankan script preprocessing:

```bash
cd preprocessing
python automate_Fauzi-Alfadhillah.py --input ../CustomerSegmentation_raw.csv --output-dir CustomerSegmentation_preprocessing
```

Output akan tersimpan di:

```text
preprocessing/CustomerSegmentation_preprocessing/customer_segmentation_preprocessed.csv
preprocessing/CustomerSegmentation_preprocessing/label_mapping.csv
```

## GitHub Actions

Workflow `.github/workflows/preprocessing.yml` akan menjalankan preprocessing otomatis ketika ada push ke branch `main` atau ketika workflow dijalankan manual melalui `workflow_dispatch`.
=======
# Eksperimen_SML_Fauzi-Alfadhillah

