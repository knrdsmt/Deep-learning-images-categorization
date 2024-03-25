import os
import shutil
import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def resize_and_pad(image, size, pad_color=0):
    h, w = image.shape[:2]
    sh, sw = size

    if h > sh or w > sw:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    aspect = w / h

    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    if len(image.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
        pad_color = [pad_color] * 3

    scaled_image = cv2.resize(image, (new_w, new_h), interpolation=interp)
    scaled_image = cv2.copyMakeBorder(scaled_image, pad_top, pad_bot, pad_left, pad_right,
                                      borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return scaled_image

def categorisation(path, folder_name):
    files = os.listdir(path)
    images = []
    for file in files:
        img_path = os.path.join(path, file)
        img = image.load_img(img_path)
        img_data = image.img_to_array(img)
        size = 128
        img_data = resize_and_pad(img_data, (size, size))

        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        images.append(img_data)

    images = np.vstack(images)
    model = VGG16(weights='imagenet', include_top=False)

    features = model.predict(images)

    features = features.reshape(features.shape[0], -1)

    pca = PCA(n_components=0.5)  # zachowaj 95% wariancji
    features_pca = pca.fit_transform(features)

    kmeans = KMeans(n_clusters=8, random_state=20).fit(features_pca)

    labels = kmeans.labels_
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=True)

    for i, file in enumerate(files):
        label = labels[i]
        directory = os.path.join(folder_name, str(label))
        if not os.path.exists(directory):
            os.makedirs(directory)
        shutil.copy(os.path.join(path, file), os.path.join(directory, file))


def outlayers(path, folder_name):
    files = os.listdir(path)
    images = []
    for file in files:
        img_path = os.path.join(path, file)
        img = image.load_img(img_path)
        img_data = image.img_to_array(img)
        size = 500
        img_data = resize_and_pad(img_data, (size, size))

        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        images.append(img_data)

    images = np.vstack(images)
    model = ResNet50(weights='imagenet', include_top=False)

    features = model.predict(images)

    features = features.reshape(features.shape[0], -1)

    kmeans = KMeans(n_clusters=2, random_state=64).fit(features)

    _, counts = np.unique(kmeans.labels_, return_counts=True)

    outlier_cluster = np.argmin(counts)

    outlier_indices = np.where(kmeans.labels_ == outlier_cluster)[0]

    if len(outlier_indices) > 8:
        distances = kmeans.transform(features[outlier_indices])[:, outlier_cluster]
        outlier_indices = outlier_indices[np.argsort(distances)[-8:]]

    os.makedirs(folder_name, exist_ok=True)

    directory = os.path.join(folder_name, 'Outlayers')

    for i in outlier_indices:
        file = files[i]
        if not os.path.exists(directory):
            os.makedirs(directory)
        shutil.copy(os.path.join(path, file), os.path.join(directory, file))

path = 'Images_dataset'
folder_name = 'Categorised'
categorisation(path, folder_name)
outlayers(path, folder_name)
print('\nCategorisation done')