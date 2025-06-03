User Guide
==========

This project offers two main scripts to run your analyses:

1. **main.py**  
   Use this script to run the traditional machine learning classifiers.

2. **CNN_main.py**  
   Use this script to train and deploy the convolutional neural network (CNN).

---

### How to Run

- To execute the machine learning classifiers, launch:

  .. code-block:: bash

      python main.py

- To work with the CNN model, launch:

  .. code-block:: bash

      python CNN_main.py

For both scripts, you can get a list of available options and required inputs by running:

.. code-block:: bash

    python main.py --help

or

.. code-block:: bash

    python CNN_main.py --help

---

### Modes of Operation

Both scripts support two modes:

- **Training mode:** train a new model using your dataset.
- **Classification mode:** use an already trained model to classify new images.

When running `main.py`, you can specify which machine learning classifier to use and choose the desired mode directly from the command line.

Similarly, `CNN_main.py` allows you to switch between training and classification modes via command-line arguments.

---

### Additional Resources

- Check the API Reference section for detailed descriptions of functions and parameters.

If you run into any issues, please refer to the Contributing section for reporting guidelines.



