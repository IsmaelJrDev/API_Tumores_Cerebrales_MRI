import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, ResNet101
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# === CONFIGURACIÓN ===
# Ajusta esto a donde tengas tu CSV y tus carpetas de imágenes
BASE_DIR = './Brain_MRI/' 
CSV_PATH = os.path.join(BASE_DIR, 'data_mask.csv')

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 2

# === 1. PREPARACIÓN DE DATOS (CORRECCIÓN DEL ERROR 111 CLASES) ===

print("🔄 Cargando y procesando CSV...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ No encuentro el archivo {CSV_PATH}. Asegúrate de tener 'data_mask.csv' en la carpeta.")

# Leer CSV
df = pd.read_csv(CSV_PATH)

# Corrección de Rutas: 
# El CSV original suele tener rutas absolutas antiguas. 
# Vamos a limpiar la ruta para que apunte a tu carpeta local actual.
# Ejemplo: '/home/viejo/dataset/TCGA_CS_4941/img.tif' -> 'TCGA_CS_4941/img.tif'

def fix_path(path):
    # Extrae las últimas dos partes de la ruta (Carpeta_Paciente/Archivo.tif)
    parts = path.split('/')
    return os.path.join(parts[-2], parts[-1])

# Aplicar corrección solo si es necesario (depende de tu CSV)
# Asumimos que 'image_path' es la columna con la ruta de la MRI
df['image_path'] = df['image_path'].apply(fix_path)

# Convertir la máscara (0 o 1) a String para que Keras entienda que es clasificación
df['mask'] = df['mask'].apply(lambda x: str(x))

print(f"✅ Dataset cargado: {len(df)} imágenes encontradas.")

# Dividir en Train y Validation
train_df, valid_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['mask'])

# Generadores
datagen = ImageDataGenerator(rescale=1./255)

print("⚙️ Creando generadores de datos...")
# Usamos flow_from_dataframe en lugar de directory
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=BASE_DIR,       # Carpeta base donde están las subcarpetas de pacientes
    x_col="image_path",       # Columna con la ruta relativa
    y_col="mask",             # Columna con la etiqueta (0 o 1)
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

valid_generator = datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory=BASE_DIR,
    x_col="image_path",
    y_col="mask",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

# === 2. MODELOS ===
#
#def build_resnet50():
#    base = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
#    base.trainable = False
#    x = layers.GlobalAveragePooling2D()(base.output)
#    x = layers.Dense(1024, activation='relu')(x)
#    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
#    model = models.Model(base.input, out, name="ResNet50")
#    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#    return model

def build_resnet101():
    base = ResNet101(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(1024, activation='relu')(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(base.input, out, name="ResNet101")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#
#def build_alexnet():
#    model = models.Sequential(name="AlexNet")
#    model.add(layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(256, 256, 3)))
#    model.add(layers.MaxPooling2D((3, 3), strides=2))
#    model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
#    model.add(layers.MaxPooling2D((3, 3), strides=2))
#    # Capa nombrada para Grad-CAM
#    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu', name='last_conv_layer'))
#    model.add(layers.Flatten())
#    model.add(layers.Dense(4096, activation='relu'))
#    model.add(layers.Dropout(0.5))
#    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
#    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#    return model
#
# === 3. ENTRENAMIENTO ===
if not os.path.exists('modelos_entrenados'):
    os.makedirs('modelos_entrenados')

models_dict = {
    #'resnet50': build_resnet50(),
    'resnet101': build_resnet101(),
    #'alexnet': build_alexnet()
}

for name, model in models_dict.items():
    print(f"\n🚀 Entrenando {name}...")
    try:
        model.fit(train_generator, epochs=EPOCHS, validation_data=valid_generator)
        model.save(f"modelos_entrenados/mri_{name}.keras")
        print(f"✅ Modelo {name} guardado con éxito.")
    except Exception as e:
        print(f"❌ Error entrenando {name}: {e}")