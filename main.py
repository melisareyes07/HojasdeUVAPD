import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Configurar TensorFlow para usar menos memoria GPU (si hay)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:
        print(f"Advertencia al configurar GPU: {e}")

app = FastAPI(
    title="Grapevine Leaf Disease Classifier API",
    description="API para clasificar enfermedades en hojas de vid usando una CNN",
    version="1.0.0"
)

# ==============================
# Configuración del modelo
# ==============================

MODEL_PATH = "grapevine_cnn_best.keras"  # nombre del archivo del modelo
IMG_SIZE = (224, 224)  # mismo tamaño que usaste en el entrenamiento

# Clases de tu modelo (en el MISMO orden que en el entrenamiento)
CLASS_NAMES = [
    "Black Rot",
    "ESCA",
    "Healthy",
    "Leaf Blight"
]

model = None  # variable global para el modelo


def load_model():
    """Cargar el modelo entrenado desde disco"""
    global model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")

    try:
        model = keras.models.load_model(MODEL_PATH)
        print("✓ Modelo de hojas de vid cargado exitosamente")
        return model
    except Exception as e:
        print(f"✗ Error al cargar el modelo: {e}")
        raise e


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesar imagen siguiendo los mismos pasos del notebook:
    1. Redimensionar a 224x224
    2. Convertir a RGB
    3. Convertir a array NumPy
    4. Agregar dimensión batch

    NOTA: la normalización (/255) está dentro del modelo (layer Rescaling),
    por lo que aquí NO se divide entre 255.
    """
    try:
        # Redimensionar
        image = image.resize(IMG_SIZE)

        # Asegurar RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # A NumPy
        img_array = np.array(image).astype("float32")

        # Agregar dimensión batch -> (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Cargar el modelo al iniciar la aplicación"""
    try:
        load_model()
    except Exception as e:
        print(f"Error al inicializar el modelo: {e}")
        # La app arranca igual, pero las predicciones fallarán


@app.get("/")
async def root():
    """Endpoint raíz informativo"""
    return {
        "message": "Grapevine Leaf Disease Classifier API",
        "status": "active",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict - POST con imagen de hoja de vid",
            "health": "/health - GET para verificar estado"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint sencillo"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tensorflow_version": tf.__version__
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predecir la enfermedad de una hoja de vid.

    Devuelve:
        - class_id: índice de la clase predicha
        - class_name: nombre de la clase
        - confidence: probabilidad máxima
        - probabilities: probabilidades por clase
    """

    # Verificar modelo
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    # Validar tipo de archivo
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        # Leer imagen
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocesar
        processed_image = preprocess_image(image)

        # Predicción
        prediction = model.predict(processed_image)

        probabilities = prediction[0].tolist()
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        # Mapear a nombre de clase
        if predicted_class < len(CLASS_NAMES):
            class_name = CLASS_NAMES[predicted_class]
        else:
            class_name = f"class_{predicted_class}"

        # Probabilidades por clase con nombres legibles
        prob_dict = {
            CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)
        }

        return {
            "class_id": predicted_class,
            "class_name": class_name,
            "confidence": confidence,
            "probabilities": prob_dict,
            "image_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "processed_shape": processed_image.shape
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
