# Enhancing-Network-Anomaly-Detection-A-Machine-Learning-Approach

## Overview

This project focuses on enhancing network anomaly detection in computer networks through a strategic combination of undersampling and oversampling techniques. The models implemented include Gradient Boosting, Random Forest, and Stacking. The undersampling and oversampling strategies contribute to a more balanced dataset, mitigating the impact of imbalanced class distribution inherent in network anomaly datasets. By leveraging these powerful machine learning techniques, the system achieves an impressive accuracy rate of 96% after hyperparameter tuning through grid search.

## Oversampling Techniques

Oversampling techniques involve increasing the number of minority class examples in the imbalanced dataset. This helps to improve model performance by balancing the class distribution. The oversampling methods used in this project are:

* **Random Oversampling:** Randomly duplicates minority class examples to increase their representation
* **ADASYN:** Generates synthetic minority class examples based on data distributions and difficulty of learning
* **SMOTE:** Creates synthetic examples by interpolating between several minority class instances 

## Undersampling Techniques  

Undersampling techniques reduce the number of majority class examples to balance the class distribution. This avoids biasing the model towards the majority class. The undersampling methods used here are:

* **Random Undersampling:** Randomly eliminates majority class examples
* **Near Miss Method:** Selects majority class examples based on their average distance to minority class examples

By combining oversampling and undersampling, the models are trained on a more balanced dataset leading to improved anomaly detection performance. The final model leverages an ensemble Stacking approach for enhanced accuracy.

## Usage

The Jupyter notebook contains the full machine learning pipeline including data preparation, model implementation and evaluation.

The final model is available as a serialized Python object file to enable further usage:

`final_model.pkl`

The model can be loaded and new data can be passed to generate anomaly predictions:

```
import pickle
model = pickle.load(open('final_model.pkl', 'rb'))
predictions = model.predict(new_data) 
```
