# FYS-STK4155 Project 2

Project 2 in FYS-STK4155: Solving classification and regression problems using neural networks.

## Project Group

- Abdullah Ali Ahmed
- Heidi Louise Xia Abrahamsen
- Ander Mathias Rønneberg
- Sindre Andre Øhman

## Repository structure

The project is divided into four folders:

- **Code**: All Python/Jupyter files used in the project.
- **Report**: The LaTeX files for the report, which describes the project and our findings.
- **Outputs**: All printed output generated from the code files.
- **Figures**: All figures made during the project, both those used in the report and those left out.
- **Handwritten notes**: Handwritten derivations.

## Code overview

- `requirment.txt`: Contains information on what packages are required to run the Python/Jypyter files and how to install them.
- `Functions.py, Functions_JAX.py, Utils.py`: Contains important functions that were reused often throughout many files, such as activaton functions and their derivatives.
- `NeuralNetworkClass.py, NeuralNetworkWrapper.py`: Contains the Neural Network itself and the Neural Network regressor class that can be used alongside SK-learn cross validation and grid search.
- `compute_gradient_test, NeuralNetwork_test.py, NeuralNetworkWrapper_test.py`: Test functions for testing the compute_gradient method, the NeuralNetwork class and the NeuralNetworkWrapper class.
- `main_regression.py`: Solutions to the regression part (tasks a–d) on the Runge function: OLS/Ridge/Lasso with polynomial features, and our own neural network with different optimizers, architectures and L1/L2 regularisation, all evaluated with 5-fold cross-validation.
- `main_classification.ipynb`: Jupyter notebook solving part e of the project, using Feed Forward Neural Network to make a classification predictor for the MNIST datasett.
