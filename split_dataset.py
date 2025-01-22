import os
import shutil
import random

# Direktori dataset asli
dataset_path = 'dataset'

# Direktori untuk train, validation, dan test
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

# Persentase data untuk training, validation, dan test
train_split = 0.7
val_split = 0.2
test_split = 0.1

# Membuat direktori untuk train, validation, dan test
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Mengambil semua gambar dalam dataset
all_images = []
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        for image_name in images:
            all_images.append((class_name, os.path.join(class_path, image_name)))

# Mengacak daftar gambar
random.shuffle(all_images)

# Tentukan jumlah gambar untuk training, validation, dan test
num_images = len(all_images)
train_count = int(train_split * num_images)
val_count = int(val_split * num_images)

# Pindahkan gambar ke direktori masing-masing dengan label di nama file
for i, (class_name, image_path) in enumerate(all_images):
    # Membuat direktori untuk setiap kelas dalam train, val, dan test
    if i < train_count:
        class_folder = os.path.join(train_dir, class_name)
    elif i < train_count + val_count:
        class_folder = os.path.join(val_dir, class_name)
    else:
        class_folder = os.path.join(test_dir, class_name)

    # Membuat folder kelas jika belum ada
    os.makedirs(class_folder, exist_ok=True)

    # Menyertakan label kelas dan index di dalam nama file
    image_name = os.path.basename(image_path)
    new_image_name = f"{class_name}_{i+1}.jpg"

    # Path tujuan untuk gambar baru
    dest_image_path = os.path.join(class_folder, new_image_name)

    # Menambahkan pengecekan dan pengecualian untuk menangani error saat menyalin
    try:
        # Menyalin gambar ke direktori tujuan
        shutil.copy(image_path, dest_image_path)
        print(f"Menyalin {image_path} ke {dest_image_path}")
    except Exception as e:
        print(f"Error saat menyalin {image_path} ke {dest_image_path}: {e}")

# Verifikasi pembagian dataset
print(f"Train directory: {train_dir}")
print(f"Validation directory: {val_dir}")
print(f"Test directory: {test_dir}")
