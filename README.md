# Bi-matrix Game Human Behavior Forecast Contest

Code for our model and training setup for Pset 1, CS236r Spring 2020.

We use a convolutional neural network to predict row player action frequencies given 3x3 matrix payoffs from a set of bimatrix games.

### Known dependencies:
```
# For running main_test.py
numpy
pandas
pytorch
tqdm  # Can not have this and comment it out on line 80

# For the entire project
matplotlib
scipy
```

## Generating predictions
To generate predictions given the `hb_test_feature.csv` file, run
```
python main_test.py
```
which loads a set of pre-trained models on the provided data in `hb_train_feature.csv` and `hb_train_truth.csv` and ensembles their predictions together. Some of the models might be overfitting on the training data; we're taking a chance here.

## Training
To train a model from scratch, run
```
python train.py 
```
*Note: currently training a model from scratch may or may not assume access to a GPU (need to double-check this).*  

More information regarding training arguments can be found in `train.py`.  

We also include a set of pretrained models in `models/` and complementary notebooks for the code and our write-up in `notebooks/'.

