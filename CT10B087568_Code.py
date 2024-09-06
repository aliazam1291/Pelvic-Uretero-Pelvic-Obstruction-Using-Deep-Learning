import os
import cv2
import numpy as np
from keras.applications import VGG16, InceptionV3, DenseNet121
from keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model

# Define the path to your dataset folder
dataset_folder = r'D:\pythonProject8\Data'  # Replace with the actual path

# Define image dimensions
image_width, image_height = 224, 224

# Initialize empty lists to store images and labels
images = []
labels = []

# Define the class labels
class_labels = ['UPJ_obstruction', 'Normal']

# Choose the base model (VGG16, InceptionV3, or DenseNet121)
base_model_name = 'VGG16'  # Change this to 'InceptionV3' or 'DenseNet121' to use a different base model

# Load the pre-trained base model without the top (fully connected) layers
if base_model_name == 'VGG16':
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3))
    preprocess_input = vgg_preprocess_input
elif base_model_name == 'InceptionV3':
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3))
    preprocess_input = inception_preprocess_input
elif base_model_name == 'DenseNet121':
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3))
    preprocess_input = densenet_preprocess_input
else:
    raise ValueError("Invalid base model name. Choose from 'VGG16', 'InceptionV3', or 'DenseNet121'.")

# Add custom convolutional layers on top of the base model
x = base_model.output
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(len(class_labels), activation='softmax')(x)

# Combine the base model and custom layers into a new model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the weights of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Loop through the class labels
for label in class_labels:
    # Get the path to the images for this class
    image_paths = [os.path.join(dataset_folder, label, img) for img in os.listdir(os.path.join(dataset_folder, label))]

    for image_path in image_paths:
        try:
            # Read and preprocess the image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_width, image_height))
            image = preprocess_input(image)  # Preprocess the image
            images.append(image)
            labels.append(class_labels.index(label))
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")

# Convert images and labels to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# Make predictions on the test set
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate F1-score
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Apply t-SNE to visualize the feature embeddings
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_test.reshape(X_test.shape[0], -1))

# Define a mapping dictionary for class labels
label_mapping = {0: 'PUJ_Obstruction', 1: 'Normal'}

# Map numerical labels to string representations
label_strings = [label_mapping[label] for label in y_test]

# Create DataFrame with string labels
df = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'label': label_strings})

# Plot t-SNE visualization
plt.figure(figsize=(10, 8))
sns.scatterplot(x='x', y='y', hue='label', palette=sns.color_palette('hsv', len(class_labels)), data=df, legend='full')
plt.title('t-SNE Visualization')
plt.show()

# Get the output of the last convolutional layer
last_conv_layer = model.get_layer('conv2d_1')
activation_model = Model(inputs=model.input, outputs=last_conv_layer.output)

# Get the activation maps for a sample image
sample_image = X_test[0]  # Change this to any sample image you want to visualize
sample_image = np.expand_dims(sample_image, axis=0)
activation_maps = activation_model.predict(sample_image)

# Visualize the activation maps
plt.figure(figsize=(16, 8))
for i in range(128):  # Assuming the last convolutional layer has 128 filters
    plt.subplot(8, 16, i + 1)
    plt.imshow(activation_maps[0, :, :, i], cmap='viridis')
    plt.axis('off')
    plt.text(0, -5, f'Filter {i}', color='white')
plt.show()
