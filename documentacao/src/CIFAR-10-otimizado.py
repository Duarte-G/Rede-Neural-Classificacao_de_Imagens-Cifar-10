import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Carregando a base de dados CIFAR-10
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizando os dados de treino e teste
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape: ", x_train.shape)
print("y_train.shape: ", y_train.shape)

# Numero de classes
K = len(set(y_train.flatten()))
print("Número de classes: ", K)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)

# Criando o modelo usando a API funcional
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

# Compilando e treinando o modelo com uma taxa de aprendizado ajustada
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinando o modelo com Data Augmentation
r = model.fit(datagen.flow(x_train, y_train, batch_size=32), 
              validation_data=(x_test, y_test), epochs=30)

# Plotando a curva de perda por iteração
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# Plotando a acurácia por iteração
plt.subplot(1, 2, 2)
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()

plt.show()

# Plotando a matriz de confusão
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # Se normalize = True, normaliza a matriz
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # Imprime a matriz
    print(cm)
    # Exibe matriz como imagem
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # Adiciona cores
    plt.colorbar()
    # Define eixos x e y
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Define o formato dos numeros da matriz
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # Exibe a matriz
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Previsões
p_test = model.predict(x_test).argmax(axis=1)
# Matriz de confusão com as previsões e respostas corretas
cm = confusion_matrix(y_test, p_test)
# Plota a matriz
plot_confusion_matrix(cm, list(range(10)))

# Rótulos das classes
labels = '''airplane
automobile
bird
cat 
deer
dog
frog
horse
ship
truck'''.split()

# Imagens classificadas incorretamente
misclassified_idx = np.where(p_test != y_test.flatten())[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title("True label: %s Predicted: %s" % (labels[y_test[i][0]], labels[p_test[i]]))
plt.show()
