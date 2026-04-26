import tensorflow as tf
import os

#insira seu código aqui

def optimize_model():
    # 1. Carregar o modelo treinado (.h5)
    print("Carregando modelo model.h5 para conversão...")
    model = tf.keras.models.load_model('model.h5')

    # 2. Configurar o conversor para TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 3. Aplicar a otimização
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 4. Converter o modelo
    print("Iniciando conversão e otimização para TFLite...")
    tflite_model = converter.convert()

    # 5. Salvar o modelo otimizado como model.tflite
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("✅ Sucesso: model.tflite gerado com otimização aplicada!")

if __name__ == "__main__":
    optimize_model()