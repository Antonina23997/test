from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.optimizers import RMSprop

model = VGG16(weights = 'imagenet')
model = Model(inputs = model.input, outputs = model.get_layer('block5_pool').output)
# Каталог с данными для обучения
train_dir = 'train'
# Каталог с данными для проверки
val_dir = 'val'
# Каталог с данными для тестирования
test_dir = 'test'
# Размеры изображения
img_width, img_height = 224, 224
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 10
# Размер мини-выборки
batch_size = 16
# Количество изображений для обучения
nb_train_samples = 5706
# Количество изображений для проверки
nb_validation_samples = 1224
# Количество изображений для тестирования
nb_test_samples = 1224

for layer in model.layers:
    layer.trainable = False

top_layer = model.output
#top_layer = GlobalAveragePooling2D()(top_layer)
top_layer = Flatten(input_shape = (7,7,512))(top_layer)
top_layer = Dense(4096, activation = 'relu')(top_layer)
top_layer = Dense(4096, activation = 'relu')(top_layer)
#top_layer = Dropout(0.35)(top_layer)
top_layer = Dense(1, activation = 'sigmoid')(top_layer)

new_model = Model(inputs = model.input, outputs = top_layer)

print(new_model.summary())

new_model.compile(loss='binary_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input, rescale = 1. / 255, rotation_range=40, width_shift_range=0.02, shear_range=0.02,height_shift_range=0.02, horizontal_flip=True)
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input, rescale = 1. / 255)
val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input, rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

new_model.fit(
    train_generator,
    steps_per_epoch=100,#nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=50,
) 

new_model.save('model_hard.h5')

