# Deep learning-based Random Noise Suppression on Seismic Data

This is a repository for my final project assignment in Machine Learning in Geophysics course at Institut Teknologi Bandung.
The objective of the final project is to implement a deep learning approach to solve geophysical issues. It is up to the students
which topic/issue they prefer to solve. However, I have a strong interest in implementing computer vision based algorithm to solve
issues in geophysics. That's why I'm trying to deploy the U-Net architecture (Ronneberger, 2015) to do denoising for random noise
in seismic data. 

I found my final project really fun and so that this project will be my independent research project that
i will develop during my free time and I will be very open to any discussion regarding this project. So, feel free to hit me up 
if you have any idea that can be useful for this project :D

## Project Overview
Seismic data plays a critical role in subsurface exploration, such as oil and gas exploration, earthquake monitoring, and 
geotechnical studies. However, seismic signals are often contaminated with random noise, which can obscure valuable subsurface information. 

Deep learning, particularly convolutional neural networks (CNNs) like U-Net, has shown great potential in enhancing 
signal-to-noise ratios while preserving intricate details. The diagram below represents the flowchart that being used in this project

![flowchart fpml drawio](https://github.com/user-attachments/assets/56637b39-2e2a-49dc-b4e3-133b94cef390)

## Data Augmentation
The data augmentation technique plays a vital role in generating diverse patches because it can highly determined whether the
model is being under or overfitted. In this peoject I'm using 10000 patches with various transforms to generate the data automatically. The figure below shows the step of data augmentation that I use in this project.
![Screenshot 2025-01-27 at 10 07 16](https://github.com/user-attachments/assets/295b23fa-41a0-4b27-a3ee-7580250ee52e)

## The U-Net Architecture Proposed
U-Net is a CNN for image segmentation, using an encoder to capture features and a decoder with skip connections to restore spatial details. It excels in medical imaging and tasks like denoising and super-resolution. (Ronneberger, 2015)

In this case, we are trying to use U-Net as the architect of the deep learning application to denoise random noise. The input and output channel will be 1 and 1 respectively since we are trying to suppress the random noise
![Screenshot 2025-01-27 at 10 07 55](https://github.com/user-attachments/assets/3dbc5666-e3ff-4809-9f5e-63cce1c0d9e1)

## Noising, Train, and Evaluate Data
![Screenshot 2025-01-27 at 10 08 35](https://github.com/user-attachments/assets/6b7406a9-3ac1-4851-b4ce-6b31c0d7a392)

## Hyperparameters and Training results
![Screenshot 2025-01-27 at 10 08 46](https://github.com/user-attachments/assets/6aed300e-ebf3-4133-a4b3-c57eb15e5a8c)

## Real Dataset
![Screenshot 2025-01-27 at 10 09 33](https://github.com/user-attachments/assets/53f511a9-1f76-401c-96db-94489c44f327)
