# DNN_inverse_2004IOT_Thailand
Data and code used for Inverse modeling of tsunami deposits at Thailand, 2004 IOT

DNN inverse model
========================

This is a code for performing inverse analysis of tsunami deposits using deep-learning neural network. The forward model fittnuss produces datasets of the thickness distribution of tsunami deposits with random initial conditions, and DNN constructed with tensorflow and keras learns the relation between initial conditions and depositional features. Then, the trained DNN model works as the inverse model for ancient or modern tsunami deposits. See details in Mitra, Naruse and Abe (2020).

---------------
Explanation of files

Forward_model_for_DNN_thai.py
the forward model for deposition from tsuanmis

Thai_DNN_inverse_model_SW_1700.ipynb
a jupyter notebook for performing the inversion

start_param_random_5000_thai_g5.csv:
teacher data. Initial conditions used for production of training datasets.

eta_5000_g6_300grid_thai_g5.csv:
training and test data produced by the forward model. This file is too large to store in GitHub, so that it is only available from Zenodo repository.

Thai_gs5.csv:
Data of 2004 Indian Ocean tsunami deposit measured at the Phra Thong island, Thailand. Volume-per-unit-area of 5 grain size classes were recorded.

config_g5_300grid_thai.ini:
Configuration file of the forward model used for production of the training datasets and inversion.
