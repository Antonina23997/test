from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator

test_dir = 'test'
model = load_model('model.h5')
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
nb_test_samples = 1224
batch_size = 16
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input, rescale = 1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_result = model.evaluate(
    test_generator,
    steps = 25
    )

print("test loss, test acc = ", test_result)