# Space Apps 2024 Seismic Detection

Welcome to my Space Apps 2024 repository!

## Challenge
**Seismic Detection Across the Solar System**

Planetary seismology missions struggle with the power requirements necessary to send continuous seismic data back to Earth. But only a fraction of this data is scientifically useful! Instead of sending back all the data collected, what if we could program a lander to distinguish signals from noise, and send back only the data we care about? Your challenge is to write a computer program to analyze real data from the Apollo missions and the Mars InSight Lander to identify seismic quakes within the noise!

## What I did

I designed, coded, trained, and evaluated a machine learning model for the task of labeling seismic data.

### Data

As part of the [Space Apps Seismic Detection Resources](https://www.spaceappschallenge.org/nasa-space-apps-2024/challenges/seismic-detection-across-the-solar-system/?tab=resources) they provide labeled training data for Mars and the Moon

[XA (1969-1977): Apollo Passive Seismic Experiments](https://www.fdsn.org/networks/detail/XA_1969/)

### Model

Convolutional Neural Network Autoencoder

Semi Supervised

### Results

Images and stats

## Run my code

### Enviroment
Use [Conda](https://www.anaconda.com/download) to set up an enviroment with the requirements.yml file. 

```bash
conda env create -f requirements.yml
```

### Data

The download_data.py file can be run to download data from [XA (1969-1977): Apollo Passive Seismic Experiments](https://www.fdsn.org/networks/detail/XA_1969/) for pretraining the CNNAutoencoder model. 

I have kept the lunar training data so that you are able to test the SeismicEventPredictor model by running the evaulate.ipynb file.

### Training Notebooks

There are two notebooks to train the two seperate parts of the model. The pretrain.ipynb file trains a new model using the CNNAutoencoder model class defined in models.py. Then, finetune.py uses the encoder layers of the model created from the pretraining.