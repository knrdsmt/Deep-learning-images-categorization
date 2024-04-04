# Image Categorization and Outlier Detection using Deep Learning

This Python script provides functionality for categorizing images based on their visual features using VGG16 and detecting outliers using ResNet50, both of which are deep learning models pre-trained on the ImageNet dataset.

## Categorized Image

   Category 1
<p align="left">
<img src="https://github.com/knrdsmt/Deep-learning-images-categorization/assets/97449172/320d5096-7a17-40d7-8459-3d631dd2b952" alt="Category 1" weight="100%"/>
</p>

---
   Category 2
<p align="left">
<img src="https://github.com/knrdsmt/Deep-learning-images-categorization/assets/97449172/16528dba-91bb-455b-bc13-3deb49952b25" height="67" alt="Category 2"/>
</p>

---
   Category 3
<p align="left">
<img src="https://github.com/knrdsmt/Deep-learning-images-categorization/assets/97449172/41107bfa-96fe-494b-a1c9-14f55c317423" weight="100%" alt="Category 3"/>
</p>

---
   Category 4
<p align="left">
<img src="https://github.com/knrdsmt/Deep-learning-images-categorization/assets/97449172/82370b9f-ce0c-4bc3-9456-3b0f727e35ea" height="75" alt="Category 4"/>
</p>

---
   Category 5
<p align="left">
<img src="https://github.com/knrdsmt/Deep-learning-images-categorization/assets/97449172/93832571-f6b6-420d-8da9-c6d8e7764a20" weight="100%" alt="Category 5"/>
</p>

---
   Category 6
<p align="left">
<img src="https://github.com/knrdsmt/Deep-learning-images-categorization/assets/97449172/104f05b8-0fac-4f5c-b196-bda4f2aa2112" weight="100%" alt="Category 6"/>
</p>

---
   Category 7
<p align="left">
<img src="https://github.com/knrdsmt/Deep-learning-images-categorization/assets/97449172/f9c0faf0-fa7b-42b1-9689-745d02917e34" weight="100%" alt="Category 7"/>
</p>

---
   Category 8
<p align="left">
<img src="https://github.com/knrdsmt/Deep-learning-images-categorization/assets/97449172/37a572ef-bbd1-49be-9108-2c8d57e68e32" height="70" alt="Category 8"/>
</p>

## Functionality
### Image Categorization

The script categorizes images into different folders based on their visual features. It follows these steps:

1. **Image Preprocessing**: Images are resized and padded to a standard size (128x128 pixels) for uniformity.
2. **Feature Extraction**: The VGG16 model is utilized to extract features from the preprocessed images.
3. **Dimensionality Reduction**: Principal Component Analysis (PCA) is applied to reduce the dimensionality of the feature vectors while preserving 95% of the variance.
4. **Clustering**: K-Means clustering is performed on the reduced feature vectors to categorize images into distinct clusters.
5. **Folder Creation**: Directories corresponding to each cluster are created, and images are moved to their respective folders.

### Outlier Detection

The script identifies outlier images based on their dissimilarity to other images. It follows these steps:

1. **Image Preprocessing**: Images are resized and padded to a standard size (500x500 pixels) for uniformity.
2. **Feature Extraction**: The ResNet50 model is used to extract features from the preprocessed images.
3. **Clustering**: K-Means clustering with 2 clusters is applied to the feature vectors.
4. **Outlier Identification**: Outlier cluster is determined as the one with fewer instances, and the indices of outlier images are retrieved.
5. **Outlier Selection**: If the number of outliers exceeds 8, the 8 most dissimilar images are selected based on their distances to the centroid of the outlier cluster.
6. **Folder Creation**: A directory named 'Outlayers' is created, and outlier images are moved to this folder.

## Usage

1. Clone the repository or download the script file.
2. Place your images in the specified directory (default: 'Images_dataset').
3. Run the script.
4. Ensure you have these dependencies installed before running the script.

   
### Dependencies

- Python 3.x
- OpenCV (cv2)
- Numpy
- Keras
- scikit-learn (sklearn)

### Note
- Ensure your images are stored in the 'Images_dataset' directory or specify the appropriate path.
- Adjust parameters such as image size, cluster count, etc., according to your requirements.
- This script assumes a basic understanding of Python programming and deep learning concepts.
