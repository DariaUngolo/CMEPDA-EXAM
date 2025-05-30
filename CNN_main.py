import sys
from pathlib import Path
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import CNN_codes.CNN_class as CNN_class
from CNN_codes.utilities import preprocess_nifti_images , split_data
import argparse

def main():

    image_folder = "C:\\Users\\daria\\OneDrive\\Desktop\\ESAME\\tutti_i_dati"
    atlas_path = "C:\\Users\\daria\\OneDrive\\Desktop\\ESAME\\lpba40.spm5.avg152T1.gm.label.nii.gz"
    roi_ids = (165, 166)

    images, labels = preprocess_nifti_images(image_folder, atlas_path, roi_ids)
    print(f"[DEBUG] images shape: {images.shape}")
    print(f"[DEBUG] labels shape: {labels.shape}")

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(images, labels)
    print(f"[DEBUG] x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"[DEBUG] x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
    print(f"[DEBUG] x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")



    shape=  (121,145,28,1)
    model = CNN_class.MyCNNModel(shape)

    model.compile_and_fit(x_train, y_train, x_val, y_val, x_test, y_test, n_epochs=10, batchsize=8)

if __name__ == '__main__':
    main()