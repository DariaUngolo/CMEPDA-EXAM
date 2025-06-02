Introduction
============

CMEPDA-EXAM is an advanced classification project aimed at distinguishing individuals with Alzheimer’s disease from healthy controls using brain imaging data. The dataset comprises 333 high-quality NIfTI MRI brain scans, providing a robust foundation for exploring neuroimaging and machine learning techniques.

To extract meaningful features, the project utilizes the BN_Atlas_246_LUT, a brain atlas located in the data folder, which segments the brain into specific Regions of Interest (ROIs). For each ROI, features such as the mean intensity and standard deviation were calculated. These features were then used as inputs for traditional machine learning models, including Random Forest (and its variants) and Support Vector Machines (SVM). The analysis revealed that ROIs associated with the hippocampus were the most critical for accurate classification, as identified through Feature Importance Ranking Extraction (FRE).

In addition to machine learning models, a Convolutional Neural Network (CNN) was implemented to leverage deep learning techniques. By focusing on MRI sections surrounding the hippocampus, the CNN reduces data complexity while maintaining high classification performance. This approach combines the precision of targeted feature extraction with the power of deep learning for automated diagnosis.

All trained models are saved in the trained_model folder, ready for deployment. This project serves as a bridge between neuroimaging and machine learning, offering tools and insights to enhance diagnostic accuracy for Alzheimer’s disease.

