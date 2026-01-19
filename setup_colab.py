import os
import shutil
from google.colab import drive

# --- CONFIGURACIÓN ---
# Rutas origen en Google Drive (Hardcoded según requerimiento)
SOURCE_SCRATCH = "/content/drive/MyDrive/LSA64_Shared_Work/Exp_20260118_1922/best_model.pth"
SOURCE_TRANSFER = "/content/drive/MyDrive/Proyectos/LSA64_Experiments/Exp_20260118_1559/best_model.pth"

# Rutas destino locales (Entorno Colab)
DEST_DIR = "checkpoints"
DEST_NOOB = os.path.join(DEST_DIR, "dummy_model.pth")
DEST_PRO = os.path.join(DEST_DIR, "best_model.pth")


def setup_environment():
    print("🚀 Iniciando Setup para LSA64 en Colab...")

    # 1. Montar Google Drive
    if not os.path.exists('/content/drive'):
        print("📂 Montando Google Drive...")
        drive.mount('/content/drive')
    else:
        print("✅ Drive ya está montado.")

    # 2. Instalar Dependencias
    print("📦 Instalando librerías necesarias...")
    os.system("pip install -q gradio mediapipe timm")

    # 3. Preparar directorios y copiar modelos
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"📁 Directorio '{DEST_DIR}' creado.")

    print("COPY: Transfiriendo modelos desde Drive (esto puede tardar un poco)...")

    # Copiar Modelo Scratch (Noob)
    if os.path.exists(SOURCE_SCRATCH):
        shutil.copy(SOURCE_SCRATCH, DEST_NOOB)
        print(f"✅ Modelo Scratch copiado a: {DEST_NOOB}")
    else:
        print(
            f"❌ ERROR: No se encontró el modelo Scratch en: {SOURCE_SCRATCH}")

    # Copiar Modelo Transfer (Pro)
    if os.path.exists(SOURCE_TRANSFER):
        shutil.copy(SOURCE_TRANSFER, DEST_PRO)
        print(f"✅ Modelo Transfer copiado a: {DEST_PRO}")
    else:
        print(
            f"❌ ERROR: No se encontró el modelo Transfer en: {SOURCE_TRANSFER}")

    print("\n🎉 Setup completado. Ahora puedes ejecutar: python src/app_colab.py")


if __name__ == "__main__":
    setup_environment()
