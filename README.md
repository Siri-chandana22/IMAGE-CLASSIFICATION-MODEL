# IMAGE-CLASSIFICATION-MODEL

COMPANY: CODTECH IT SOLUTIONS

NAME: YARLAGADDA SIRI CHANDANA

INTERN ID: CT06DL1154

DOMAIN: MACHINE LEARNING

DURATION: 6 WEEKS

MENTOR: NEELA SANTOSH

DESCRIPTION:

This implements an image classification model using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. The goal is to build, train, evaluate, and deploy a deep learning model that can classify color images into 10 predefined categories. The categories include common objects such as airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. This implementation also includes the functionality to make predictions on external images provided by the user.

Importing Required Libraries:

The code begins by importing necessary libraries:
NumPy for array operations and numerical computing.
Matplotlib and Seaborn for visualization.
TensorFlow/Keras for building and training the neural network.
OpenCV for image loading and preprocessing.
Scikit-learn for performance metrics like confusion matrix and classification report.
These libraries together offer a complete ecosystem for data loading, model training, visualization, and evaluation.

Loading and Normalizing the CIFAR-10 Dataset:

The CIFAR-10 dataset is directly loaded from TensorFlow's datasets module. It contains 60,000 32x32 pixel color images, split into 50,000 for training and 10,000 for testing. Each image has a corresponding label from the 10 classes. The pixel values are scaled from the original range [0, 255] to [0, 1] using normalization, which is critical for efficient training of deep learning models. Labels are also flattened from a 2D array to a 1D array to match Keras expectations.

Visualizing the Dataset:

To get an overview of the data, the first 100 training images are plotted in a 10x10 grid. Each image is displayed along with its corresponding class name. This helps in understanding the variety and quality of data and verifying that it has been loaded correctly.

Building the CNN Model:

The CNN model is constructed using TensorFlow's Keras Sequential API. The architecture includes:
Three Conv2D layers to extract spatial features.
Two MaxPooling2D layers to downsample the feature maps, reducing computational complexity and controlling overfitting.
A Flatten layer to convert the 2D output into a 1D vector.
A Dense layer with 128 neurons for fully connected processing.
A Dropout layer with a 50% dropout rate to prevent overfitting.
A final Dense layer with 10 outputs and a softmax activation for multi-class classification.
The model is compiled using the Adam optimizer, with a loss function of sparse_categorical_crossentropy, and it tracks accuracy as the performance metric.

Training the Model:

The model is trained for 10 epochs with a batch size of 64. During training, validation is performed on the test set to monitor overfitting and generalization. The training progress, including loss and accuracy, is tracked and stored in the history variable.

Evaluating the Model:

After training, the model's performance is evaluated on the test set. The overall test accuracy is printed. Predictions are made using the model, and the highest probability class for each image is selected using argmax.
A confusion matrix is generated and visualized using Seaborn, showing the counts of true vs. predicted labels for each class. This is useful to identify where the model is performing well or making systematic mistakes.
A classification report is also printed, including precision, recall, F1-score, and support for each class. This provides a detailed quantitative assessment of the model's performance.

Saving the Model:

The trained model is saved to disk in .keras format using:
python
Copy
Edit
model.save("cnn_cifar10_model.keras")
This allows you to reload the model later without retraining, which is useful for deployment or further testing.

Predicting External Images:

The script defines a function predict_external_image(image_path) which allows users to test the model on their own images:
It reads the image using OpenCV.
Converts it from BGR to RGB.
Resizes it to 32x32 to match the CIFAR-10 input size.
Normalizes and reshapes the image.
Passes it through the model to get predictions.
Displays the image along with the predicted class and confidence score.
This functionality simulates a real-world scenario where a trained model is used to classify unseen images.

Conclusion:

This project demonstrates an end-to-end workflow for image classification using deep learning:
From loading and exploring data,
Building and training a CNN model,
Evaluating performance,
Visualizing results, and
Deploying the model for external image predictions.

OUTPUT:

DATASET IMAGE:

