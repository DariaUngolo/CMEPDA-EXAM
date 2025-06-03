Introduction
============

Overview
--------
This project focuses on the development and implementation of a **binary classifier** aimed at distinguishing between:

- Subjects diagnosed with **Alzheimer’s Disease (AD)**.
- **Healthy control subjects (CTRL)**.

The primary goal is to leverage **3D brain MRI scans** to build a robust classification model for early diagnosis and better understanding of Alzheimer’s Disease.

Dataset Description
-------------------
The dataset used in this project consists of:

- A total of **333 subjects**, including:
  - **144 patients with AD**.
  - **189 healthy controls**.

- **3D brain MRI images** in NIfTI format.

- **Two brain atlases**, which parcellate the brain into anatomically meaningful **Regions of Interest (ROIs)**:
  - Atlas 1: Divides the brain into **56 ROIs**.
  - Atlas 2: Divides the brain into **246 ROIs**, providing higher spatial resolution.

Key Components
--------------
- **Look-Up Tables (LUTs)**:
  Each atlas is accompanied by a LUT, listing:
  - The names of all ROIs.
  - Their corresponding integer labels.

These tables are crucial for interpreting the ROIs during feature extraction and analysis.

- **Regions of Interest (ROIs)**:
  The ROIs provide anatomically meaningful divisions of the brain, essential for extracting relevant features.

Conclusion
----------
The project aims to utilize this rich dataset and associated tools to create a highly accurate classifier, offering insights into the neuroanatomical distinctions between Alzheimer’s patients and healthy controls.
