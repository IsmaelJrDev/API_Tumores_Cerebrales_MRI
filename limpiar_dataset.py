import os
from PIL import Image

# --- CONFIGURACIÓN ---
# Asegúrate de que el nombre de la carpeta sea EXACTAMENTE el que tienes.
# Si tu carpeta se llama 'Brain_MRI', déjalo así.
DATA_DIR = './Brain_MRI/' 

def verificar_y_limpiar():
    # Verificar si la carpeta existe antes de empezar
    if not os.path.exists(DATA_DIR):
        print(f"❌ ERROR: No encuentro la carpeta '{DATA_DIR}'.")
        print("   Asegúrate de estar ejecutando este script en la misma carpeta donde está 'Brain_MRI'.")
        print(f"   Tu ubicación actual es: {os.getcwd()}")
        return

    print(f"🔍 Escaneando directorio: {DATA_DIR} ...")
    archivos_corruptos = 0
    archivos_revisados = 0
    
    # Recorrer todas las carpetas y subcarpetas
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            archivos_revisados += 1
            
            try:
                # Intentamos abrir la imagen con Pillow
                with Image.open(file_path) as img:
                    img.verify() # Verifica si el archivo está roto internamente
            except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
                print(f"❌ Archivo corrupto detectado y ELIMINADO: {file_path}")
                try:
                    os.remove(file_path)
                    archivos_corruptos += 1
                except Exception as del_err:
                    print(f"   No se pudo eliminar: {del_err}")

    print("-" * 30)
    print(f"📊 Total revisados: {archivos_revisados}")
    if archivos_corruptos == 0:
        print("✅ No se encontraron archivos corruptos. ¡Tu dataset está limpio!")
    else:
        print(f"🧹 Se eliminaron {archivos_corruptos} archivos corruptos.")

if __name__ == "__main__":
    verificar_y_limpiar()