import shutil
import os

# Каталоги с набором данных
data_dir_noorigami = 'C:\\origami_classification_data\\no_origami'
data_dir_origami = 'C:\\origami_classification_data\\origamiYFCC'
# Каталог с данными для обучения
train_dir = 'train'
# Каталог с данными для проверки
val_dir = 'val'
# Каталог с данными для тестирования
test_dir = 'test'
# Часть набора данных для тестирования
test_data_portion = 0.15
# Часть набора данных для проверки
val_data_portion = 0.15
# Количество элементов данных в одном классе
nb_images = 4000
i, j = 0, 0 

#for file in os.listdir(data_dir_noorigami):
#    os.rename(os.path.join(data_dir_noorigami, file), os.path.join(data_dir_noorigami, f'no_origami.{i}.jpg'))
#    i+=1

#for file in os.listdir(data_dir_origami):
#    os.rename(os.path.join(data_dir_origami, file), os.path.join(data_dir_origami, f'origami.{j}.jpg'))
#    j+=1

#def create_directory(dir_name):
#    if os.path.exists(dir_name):
#        shutil.rmtree(dir_name)
#    os.makedirs(dir_name)
#    os.makedirs(os.path.join(dir_name, "origami"))
#    os.makedirs(os.path.join(dir_name, "no_origami"))

#create_directory(train_dir)
#create_directory(val_dir)
#create_directory(test_dir)

def copy_images(start_index, end_index, dest_dir):
    for i in range(start_index, end_index):
        shutil.copy2(os.path.join(data_dir_origami, "origami." + str(i) + ".jpg"),
                    os.path.join(dest_dir, "origami"))
        shutil.copy2(os.path.join(data_dir_noorigami, "no_origami." + str(i) + ".jpg"),
                   os.path.join(dest_dir, "no_origami"))

start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
start_test_data_idx = int(nb_images * (1 - test_data_portion))
print(start_val_data_idx)
print(start_test_data_idx)

copy_images(0, start_val_data_idx, train_dir)
copy_images(start_val_data_idx, start_test_data_idx, val_dir)
copy_images(start_test_data_idx, nb_images, test_dir)

print("khuy")
