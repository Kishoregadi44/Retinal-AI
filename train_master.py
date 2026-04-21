import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, applications
import os

dataset_path = 'dataset/train'

# 1. Advanced Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_path, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_path, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='validation'
)

# 2. LOAD PRE-TRAINED BRAIN (MobileNetV2)
# We freeze the base layers so we don't destroy Google's knowledge
base_model = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

# 3. Add your "Medical Specialty" layers on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax') # 4 Categories
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train - This will be MUCH more accurate
print("--- Training the Genius Model (Transfer Learning) ---")
model.fit(train_gen, validation_data=val_gen, epochs=10)

# 5. Save the updated Brain
if not os.path.exists('models'): os.makedirs('models')
model.save('models/diabetic_retinopathy_v1.h5')
print("--- SUCCESS: High-Accuracy Model Saved ---")
