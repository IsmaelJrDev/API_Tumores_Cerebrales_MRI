import numpy as np
import tensorflow as tf
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Crear un sub-modelo que vaya desde la entrada hasta la última capa convolucional
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Grabar las operaciones para calcular el gradiente
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. Calcular gradientes de la clase predicha hacia el mapa de características
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 4. Multiplicar cada canal por su importancia
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 5. Normalizar el mapa de calor entre 0 y 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, output_path_mask, output_path_overlay, alpha=0.4):
    # Cargar imagen original
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256)) # TAMAÑO IMPORTANTE: Debe coincidir con tu entrenamiento (256x256)
    
    # Escalar el heatmap al tamaño de la imagen (0-255)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Colorear el heatmap (Rojo = Caliente)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Crear superposición
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    
    # Guardar archivos
    cv2.imwrite(output_path_mask, heatmap_color)
    cv2.imwrite(output_path_overlay, overlay)