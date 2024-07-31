# Ship detection

This repository contains links to Jupyter notebooks on Kaggle, which cover exploratory data analysis (EDA) and model training/inference using TensorFlow U-Net. The training and inference were conducted on Kaggle with a P100 GPU.

## Notebooks

1. **Exploratory Data Analysis (EDA) of the Dataset**
   - [Link to the Notebook](https://www.kaggle.com/code/mykoladzhun/eda-of-the-dataset)
   - This notebook provides an in-depth analysis of the dataset, exploring various features and distributions.
   - **File:** `eda-of-the-dataset.ipynb`

2. **Training TensorFlow U-Net (Dice Coefficient: 0.46)**
   - [Link to the Notebook](https://www.kaggle.com/code/mykoladzhun/tensorflow-u-net-dice-0-46)
   - This notebook demonstrates the training process of a U-Net model using TensorFlow. The model achieved a Dice coefficient of 0.46.
   - **File:** `tensorflow-u-net-dice-0-46.ipynb`
   - **Model Weights:** `unet_model_sr08_dc_046.weights.h5`

3. **Inference with TensorFlow U-Net**
   - [Link to the Notebook](https://www.kaggle.com/code/mykoladzhun/inference-tensorflow-u-net)
   - This notebook covers the inference phase using the trained U-Net model to make predictions on new data.
   - **File:** `inference-tensorflow-u-net.ipynb`

## Environment

- **Training and Inference** were performed on Kaggle, utilizing a P100 GPU.

## Potential Improvements

To further improve the results, the following strategies could be considered:

1. **Change the Model Architecture:** Use a more complex model architecture to better capture the nuances in the data.
2. **Ensemble Multiple Architectures:** Combine predictions from different model architectures to improve overall performance.
3. **Pseudo Labeling:** Use pseudo labeling to incorporate predictions on the test data into the training process.
4. **Data Augmentation:** Apply various data augmentation techniques to increase the diversity of the training dataset.
5. **Use a Pretrained Model:** Leverage a pretrained model on a similar task to potentially improve the performance.
