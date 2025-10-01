# EuroSAT Satellite Image Classification & Analysis for Poverty Alleviation SDG goal 🛰️

This project demonstrates a complete workflow for classifying satellite images from the EuroSAT dataset using a deep learning model. It employs transfer learning with a pre-trained ResNetV2 architecture, evaluates the model's performance, and extends the analysis with Explainable AI (XAI) using Grad-CAM and a custom "Economic Opportunity Map" application.

## 🌟 Key Features

* **Image Classification:** Classifies satellite images into 10 distinct land use/land cover categories.
* **Transfer Learning:** Utilizes a pre-trained **ResNet50V2** model for efficient and powerful feature extraction.
* **High Performance:** Achieves an overall test accuracy of **94%**.
* **Efficient Data Pipeline:** Implements an optimized `tf.data` pipeline for fast data loading and preprocessing.
* **Overfitting Prevention:** Uses **Early Stopping** to halt training when the model's performance on validation data plateaus.
* **Explainable AI (XAI):** Integrates **Grad-CAM** to visualize the regions of an image the model focuses on for its predictions.
* **Practical Application:** Maps land use predictions to an "Economic Opportunity Score" to identify areas with potential for development.

---

## Dataset 📂

The project uses the **EuroSAT dataset**, which consists of 27,000 labeled satellite images acquired by the Sentinel-2 satellite. The images are categorized into 10 classes:

* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake

Each class contains 2,000 to 3,000 images of size 64x64 pixels.

---

## 🛠️ Methodology

The project follows a structured machine learning pipeline:

### 1. Data Preparation & Preprocessing

The dataset is loaded and organized into a Pandas DataFrame. It's then split into a training set (80%) and a testing set (20%). An efficient `tf.data` pipeline is created to:
1.  Read image files.
2.  Decode them as JPEGs.
3.  Resize all images to `224x224` pixels to match the model's input shape.
4.  Apply `ResNetV2`-specific preprocessing.
5.  Batch the data and prefetch batches for optimal GPU utilization.

### 2. Model Architecture (Transfer Learning)

The model leverages transfer learning to achieve high accuracy with less training time.

* **Base Model:** A **ResNet50V2** model, pre-trained on the ImageNet dataset, is used as the feature extractor. Its weights are frozen (`trainable = False`) to retain the learned features.
* **Custom Head:** A custom classification head is added on top of the base model:
    1.  `GlobalAveragePooling2D`: Flattens the feature maps from the base model.
    2.  `Dense (128 units, ReLU)`: An intermediate dense layer for learning higher-level features.
    3.  `Dense (10 units, Softmax)`: The final output layer that produces probabilities for the 10 classes.

### 3. Model Training

* **Compiler:** The model is compiled using the `adam` optimizer, `sparse_categorical_crossentropy` loss function, and `accuracy` as the evaluation metric.
* **Callbacks:** `EarlyStopping` is used to monitor `val_accuracy` with a patience of 3 epochs. This prevents overfitting by stopping the training process when performance on the test set stops improving and restoring the model weights from the best epoch.

---

## 📊 Results & Evaluation

The model was trained for 7 epochs before early stopping was triggered, restoring the best weights from epoch 4. It achieved a final **test accuracy of 94%**.

### Training History

The accuracy and loss curves show that the model learned effectively and that early stopping successfully prevented overfitting.


### Classification Report

The detailed classification report shows excellent performance across all classes, with high precision, recall, and F1-scores.

```
--- Classification Report ---
                      precision    recall  f1-score   support

          AnnualCrop       0.97      0.90      0.93       600
              Forest       0.95      0.98      0.97       600
HerbaceousVegetation       0.92      0.95      0.93       600
             Highway       0.91      0.92      0.91       500
          Industrial       0.97      0.96      0.97       500
             Pasture       0.92      0.91      0.91       400
       PermanentCrop       0.89      0.93      0.91       500
         Residential       0.96      0.99      0.97       600
               River       0.94      0.90      0.92       500
             SeaLake       0.99      0.97      0.98       600

            accuracy                           0.94      5400
           macro avg       0.94      0.94      0.94      5400
        weighted avg       0.94      0.94      0.94      5400
```

---

## 🧠 Explainable AI (XAI) with Grad-CAM

To understand the model's decision-making process, **Gradient-weighted Class Activation Mapping (Grad-CAM)** is used. This technique produces a heatmap that highlights the most important regions in an image for a given prediction.

The example below shows that for a "River" prediction, the model correctly focuses its attention on the river itself.


---

## 💡 Application: Economic Opportunity Map

This project introduces a practical application by translating classification results into an **Economic Opportunity Map**. Each land use class is assigned a score based on its potential for economic development.

**Scoring System:**
* **High Opportunity (Score: 3):** `Industrial`
* **Moderate Opportunity (Score: 2):** `Highway`
* **Some Opportunity (Score: 1):** `Residential`, `PermanentCrop`
* **Neutral/Agricultural (Score: 0):** `AnnualCrop`, `Pasture`
* **Low Opportunity/Protected (Score: -1):** `HerbaceousVegetation`, `Forest`, `River`, `SeaLake`

The map below visualizes a sample of test images, color-coded by their opportunity score, providing a quick visual reference for land assessment.


---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://your-repository-url.git](https://your-repository-url.git)
    cd your-repository-directory
    ```

2.  **Install the required libraries:**
    ```bash
    pip install tensorflow matplotlib scikit-learn pandas
    ```

3.  **Download the Dataset:**
    Download the EuroSAT dataset and place it in a directory accessible to the notebook. Update the `DATA_PATH` variable in the notebook if necessary.
    * Dataset link: [EuroSAT Dataset](https://github.com/phelber/EuroSAT)

4.  **Run the Jupyter Notebook:**
    Launch Jupyter Notebook and open `datascience.ipynb`.
    ```bash
    jupyter notebook datascience.ipynb
    ```

---

## 💻 Technologies Used

* **Python 3**
* **TensorFlow & Keras:** For building and training the deep learning model.
* **Scikit-learn:** For data splitting and model evaluation (classification report).
* **Pandas:** For data manipulation and management.
* **NumPy:** For numerical operations.
* **Matplotlib & OpenCV:** For image processing and visualization.
