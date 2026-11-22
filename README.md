# SE313-RecognitionNumber
Recognition written number with ANN
## How to use:
* First clone this repo and create Python environment with ```.venv```
* Move to the folder project and use ``` .venv\Scripts\Activate.ps1 ``` to create virtual Python environment
* Use ```python test_gui.py``` to experience the recognition
## Description:
* The ```ann_model.py``` is the model ANN create from scratch
* The ```train.py``` is how to train model of MNIST dataset with saving checkpoint and ann_weight
* The ```evaluate_test.py``` to evaluate with the training data of the MNIST dataset
* The ```evaluate_usps.py``` to evaluate with the training data of the USPS dataset
* The ```test_gui.py``` to recognize number
