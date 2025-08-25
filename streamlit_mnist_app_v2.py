import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Configuración de la página
st.set_page_config(page_title="Clasificador de Dígitos", layout="centered")

# Título
st.title("✍️ Clasificador de Dígitos MNIST")
st.markdown("Dibuja un número del 0 al 9 y la red neuronal intentará adivinar cuál es.")

# Cache para cargar el modelo una sola vez
@st.cache_resource
def load_model():
    """Carga el modelo Keras entrenado desde el disco."""
    try:
        model = keras.models.load_model('modelo_mnist_diplomado.h5')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.warning("Asegúrate de que el archivo 'modelo_mnist_diplomado.h5' esté en el mismo directorio que este script.")
        return None

def preprocess_image(canvas_data):
    """
    Convierte el dibujo del canvas al formato MNIST:
    - Gris ('L'), dígito blanco sobre fondo negro
    - Escalado manteniendo aspecto: lado mayor = 20 px
    - Pegado en lienzo 28x28 y centrado por centroide
    - Normalizado [0,1] y aplanado (1, 784)
    """
    if canvas_data is None or canvas_data.image_data is None:
        return None

    # 1) RGBA -> L (grises)
    img = Image.fromarray(canvas_data.image_data.astype("uint8"), "RGBA").convert("L")
    # 2) invertir: fondo negro (0), dígito blanco (255)
    img = ImageOps.invert(img)

    # 3) recortar al contenido
    bbox = img.getbbox()
    if bbox is None:
        return None
    img = img.crop(bbox)

    # 4) escalar manteniendo aspecto: lado mayor = 20 px (como MNIST)
    w, h = img.size
    if w > h:
        new_w, new_h = 20, max(1, int(round(h * (20.0 / w))))
    else:
        new_h, new_w = 20, max(1, int(round(w * (20.0 / h))))
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    img = img.resize((new_w, new_h), resample)

    # 5) pegar en lienzo 28x28 negro, centrado
    canvas = Image.new("L", (28, 28), color=0)
    offset = ((28 - new_w) // 2, (28 - new_h) // 2)
    canvas.paste(img, offset)

    # 6) centrar por centroide (como en el preproc clásico de MNIST)
    arr = np.array(canvas).astype("float32")
    s = arr.sum()
    if s > 0:
        # centroide (cx, cy)
        ys = np.arange(28, dtype=np.float32)
        xs = np.arange(28, dtype=np.float32)
        cy = (arr.sum(axis=1) * ys).sum() / s
        cx = (arr.sum(axis=0) * xs).sum() / s
        shift_x = int(round(14 - cx))
        shift_y = int(round(14 - cy))

        # shift con roll y rellenar bordes con 0
        arr = np.roll(arr, shift_x, axis=1)
        arr = np.roll(arr, shift_y, axis=0)
        if shift_x > 0:   arr[:, :shift_x] = 0
        elif shift_x < 0: arr[:, shift_x:] = 0
        if shift_y > 0:   arr[:shift_y, :] = 0
        elif shift_y < 0: arr[shift_y:, :] = 0

    # 7) normalizar y aplanar
    arr = (arr / 255.0).astype("float32").reshape(1, 784)
    return arr




# Cargar el modelo
model = load_model()

if model:
    st.subheader("Dibuja un dígito aquí 👇")
    
    # Crear un canvas para dibujar
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=12,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    # Botones de acción
    col1, col2 = st.columns(2)
    with col1:
        predict_button = st.button("Predecir", use_container_width=True, type="primary")
    with col2:
        clear_button = st.button("Limpiar", use_container_width=True)

    # Lógica de predicción
    if predict_button and canvas_result.image_data is not None:
        processed_img = preprocess_image(canvas_result)
        
        if processed_img is not None:
            predictions = model.predict(processed_img, verbose=0)
            predicted_digit = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            st.header("Resultado de la Predicción")
            st.metric("El modelo predice que es un:", f"{predicted_digit}", f"Confianza: {confidence:.2%}")
            
            st.subheader("Probabilidades por Dígito")
            fig, ax = plt.subplots(figsize=(10, 5))
            digits = list(range(10))
            probabilities = predictions[0]
            
            bars = ax.bar(digits, probabilities, color='skyblue')
            bars[predicted_digit].set_color('royalblue')
            
            ax.set_xlabel('Dígito')
            ax.set_ylabel('Probabilidad')
            ax.set_title('Distribución de Probabilidades')
            ax.set_xticks(digits)
            ax.set_ylim(0, 1)
            
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2%}', ha='center', va='bottom')

            st.pyplot(fig)
        else:
            st.warning("⚠️ Por favor, dibuja un dígito antes de predecir.")

    if clear_button:
        st.rerun()

    with st.expander("Ver instrucciones de uso"):
        st.write("""
        1.  **Dibuja un solo dígito** (del 0 al 9) en el centro del recuadro blanco.
        2.  Intenta hacerlo de un solo trazo y con un tamaño razonable.
        3.  Haz clic en **'Predecir'**.
        4.  Usa **'Limpiar'** para borrar y empezar de nuevo.
        """)
else:
    st.error("La aplicación no puede funcionar sin el modelo. Verifica que el archivo del modelo esté disponible.")