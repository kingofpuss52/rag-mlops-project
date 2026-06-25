#!/bin/bash

# Hapus sisa database lama agar bersih total
rm -rf data/vector_db
rm -f optuna_study.db

echo "🚀 Memulai Bayesian Optimization Loop via CLI..."

# Lakukan looping sebanyak 5 trial secara otomatis dan terisolasi murni
for i in {1..20}
do
   echo -e "\n----------------------------------------"
   echo "▶️ MENJALANKAN EXPERIMEN OPTUNA TRIAL $i"
   echo "----------------------------------------"
   
   # Panggil master tuning untuk menentukan parameter dan eksekusi
   python tuning_master.py
   
   # Bersihkan folder database vektor secara paksa tepat setelah proses Python mati
   rm -rf data/vector_db
   echo "🧹 Database vektor berhasil dibersihkan. Siap ke trial berikutnya."
done

echo -e "\n🎯 Rangkaian Bayesian Optimization selesai tanpa tabrakan file!"