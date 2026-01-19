from mvit import LSA64MViT
import gradio as gr
import torch
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from torchvision.transforms import v2
import sys
import os

# Aseguramos que Python encuentre el módulo mvit si se ejecuta desde root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- CONFIGURACIÓN DE RENDIMIENTO (COLAB) ---
# Detectamos GPU automáticamente
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Dispositivo de inferencia: {DEVICE}")

SKIP_FRAMES = 3    # Reducimos el salto porque la GPU es más rápida
DISPLAY_W, DISPLAY_H = 640, 480

# Configuración del Modelo
NUM_FRAMES = 16
IMG_SIZE = 224
# Rutas relativas a la raíz del repo
PATH_MODEL_PRO = "checkpoints/best_model.pth"
PATH_MODEL_NOOB = "checkpoints/dummy_model.pth"

# Diccionario LSA64
LSA64_MAP = {
    0: "Opaquear", 1: "Rojo", 2: "Verde", 3: "Amarillo", 4: "Brillante",
    5: "Celeste", 6: "Colores", 7: "Rosa", 8: "Mujer", 9: "Enemigo",
    10: "Hijo", 11: "Hombre", 12: "Lejos", 13: "Cajón", 14: "Nacer",
    15: "Aprender", 16: "Llamar", 17: "Mofeta", 18: "Amargo", 19: "Leche",
    20: "Agua", 21: "Comida", 22: "Argentina", 23: "Uruguay", 24: "País",
    25: "Apellido", 26: "Dónde", 27: "Burlarse", 28: "Cumpleaños", 29: "Desayuno",
    30: "Foto", 31: "Hambre", 32: "Mapa", 33: "Moneda", 34: "Música",
    35: "Nave", 36: "Ninguno", 37: "Nombre", 38: "Paciencia", 39: "Perfume",
    40: "Sordo", 41: "Trampa", 42: "Arroz", 43: "Asado", 44: "Caramelos",
    45: "Chicle", 46: "Fideos", 47: "Yogur", 48: "Aceptar", 49: "Agradecer",
    50: "Apagar", 51: "Aparecer", 52: "Aterrizar", 53: "Atrapar", 54: "Ayudar",
    55: "Bailar", 56: "Bañarse", 57: "Comprar", 58: "Copiar", 59: "Correr",
    60: "Darse cuenta", 61: "Dar", 62: "Encontrar", 63: "Saltar/Otro"
}

# --- SETUP MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1  # Subimos complejidad ya que estamos en Colab
)

# --- CARGA DE MODELOS ---


def load_model(path, is_scratch=False):
    print(f"⚙️ Cargando {path} en {DEVICE}...")
    weights = None if is_scratch else "DEFAULT"
    model = LSA64MViT(num_classes=64, weights=weights)

    if not is_scratch:
        try:
            checkpoint = torch.load(path, map_location=DEVICE)
            sd = checkpoint['model_state_dict'] if isinstance(
                checkpoint, dict) else checkpoint
            model.load_state_dict(sd)
            print("   ✅ Pesos cargados correctamente.")
        except Exception as e:
            print(f"   ⚠️ Error cargando pesos: {e}")

    model.to(DEVICE)
    model.eval()
    # NOTA: Eliminamos quantization_dynamic para permitir uso de GPU
    return model


model_pro = load_model(PATH_MODEL_PRO, is_scratch=False)
model_noob = load_model(PATH_MODEL_NOOB, is_scratch=True)

# --- PROCESAMIENTO ---
buffer = deque(maxlen=NUM_FRAMES)
transform = v2.Compose([
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Resize(256, antialias=True), v2.CenterCrop(IMG_SIZE),
    v2.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

last_pro = {"Iniciando...": 0.0}
last_noob = {"Iniciando...": 0.0}
frame_count = 0


def process_frame(image):
    global frame_count, last_pro, last_noob
    if image is None:
        return image, {}, {}

    # Resize para visualización y MediaPipe
    image_small = cv2.resize(image, (DISPLAY_W, DISPLAY_H))

    # MediaPipe (CPU bound, pero rápido en resolución baja)
    results = holistic.process(image_small)
    annotated = image_small.copy()

    mp_drawing.draw_landmarks(
        annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
    mp_drawing.draw_landmarks(
        annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

    # Inferencia
    buffer.append(image_small)

    if len(buffer) == NUM_FRAMES and frame_count % SKIP_FRAMES == 0:
        # Preparamos tensor y enviamos a GPU
        tensor = torch.stack([torch.from_numpy(f) for f in buffer])
        tensor = transform(tensor.permute(0, 3, 1, 2)).permute(
            1, 0, 2, 3).unsqueeze(0)
        tensor = tensor.to(DEVICE)

        with torch.no_grad():
            # PRO
            probs_pro = torch.nn.functional.softmax(
                model_pro(tensor), dim=1).squeeze()
            top_pro = torch.topk(probs_pro, 3)
            last_pro = {LSA64_MAP.get(i.item(), str(i.item())): float(
                v) for i, v in zip(top_pro.indices, top_pro.values)}

            # NOOB
            probs_noob = torch.nn.functional.softmax(
                model_noob(tensor), dim=1).squeeze()
            top_noob = torch.topk(probs_noob, 3)
            last_noob = {LSA64_MAP.get(i.item(), str(i.item())): float(
                v) for i, v in zip(top_noob.indices, top_noob.values)}

    frame_count += 1
    return annotated, last_pro, last_noob


# --- INTERFAZ ---
css = ".gradio-container {background-color: #0f172a}"

with gr.Blocks(title="LSA64 Colab GPU", css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 LSA64: Cloud GPU Demo")
    with gr.Row():
        video_display = gr.Image(
            sources=["webcam"], streaming=True, label="Webcam Stream")
        with gr.Column():
            lbl_pro = gr.Label(num_top_classes=3, label="Experto (Transfer)")
            lbl_noob = gr.Label(num_top_classes=3, label="Novato (Scratch)")

    video_display.stream(process_frame, inputs=video_display, outputs=[video_display, lbl_pro, lbl_noob],
                         stream_every=0.05, time_limit=300)

if __name__ == "__main__":
    # share=True genera el link público (ej: https://xxxx.gradio.live)
    demo.launch(share=True, debug=True)
