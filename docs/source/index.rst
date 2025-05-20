.. CMEPDA-EXAM documentation master file, created by
   sphinx-quickstart on Sat May 17 16:12:35 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CMEPDA-EXAM Documentation
==========================

Welcome to the documentation for CMEPDA-EXAM. 

Contents Index:
---------------

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   Introduction
   Installation
   User Guide
   API Reference
   Contributing


Introduction
============

CMEPDA-EXAM is a classification project aimed at distinguishing subjects with Alzheimer’s disease from those without, using brain imaging data.
The dataset consists of 333 NIfTI brain scans. 
The project utilizes the BN_Atlas_246_LUT, a brain atlas that divides the brain into distinct regions of interest (ROIs), to extract meaningful features for classification.

The implemented classification models include Random Forest and its variants, as well as Support Vector Machines (SVM).
 This project combines neuroimaging data processing with machine learning techniques to improve diagnostic accuracy.


