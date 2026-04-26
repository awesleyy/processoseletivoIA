import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#insira seu código aqui

def train_model():
    # 1. Carregar o dataset MNIST
    print("Carregando dataset MNIST...")
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. Pré-processamento
    # Normalizar os pixels para o intervalo [0, 1] e redimensionar para (28, 28, 1)
    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

    # 3. Construção do Modelo CNN
    # Usando Conv2D e MaxPooling conforme os requisitos obrigatórios
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # 4. Compilação do Modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. Treinamento
    # Limitado a 5 épocas para ser compatível com o ambiente de CI
    print("Iniciando treinamento...")
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

    # 6. Avaliação Final e Salvamento
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nAcurácia final no teste: {acc:.4f}")

    # Salva o modelo no formato Keras (.h5)
    model.save('model.h5')
    print("✅ Modelo salvo com sucesso como 'model.h5'")

if __name__ == "__main__":
    train_model()