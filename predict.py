from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import os

# Muat mode
test_path = os.path.join('.', 'dataset', 'test')
model = load_model("durian_leaf_classifier.h5")

# Muat class_indices dari file JSON
with open('class_indices.json', 'r') as json_file:
    class_indices = json.load(json_file)

# Membalik mapping
class_labels = {v: k for k, v in class_indices.items()}

# Path ke gambar baru
image_path = os.path.join(test_path,'bawor','bawor_1463.jpg')

# Preprocessing gambar
img_width, img_height = 150, 150
image = load_img(image_path, target_size=(img_width, img_height))
image_array = img_to_array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Prediksi
predictions = model.predict(image_array)
predicted_class = np.argmax(predictions, axis=1)

# Hasil prediksi
predicted_label = class_labels[predicted_class[0]]
print(f"Prediksi kelas: {predicted_label}")
