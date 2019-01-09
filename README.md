# Speech Enhancement using Deep Neural Networks
## Introduction
Whenever we work with real time speech signals, we need to keep in mind about various types of noises that gets added and hence resulting in corruption of noise.
Therefore, in order to make a better sense of the signals, it is very much necessary to enhance the speech signals by removing the noises present in them. 

## Applications:
* Automatic speech recognition
* Speaker recognition
* Mobile communication
* Hearing aids 

## DNN based architectures
* Autoencoder Decoder 
* Recurrent Neural Nets
* Restricted Boltzmann Machines

## Dataset:
The dataset used for this project is TCD-TIMIT speech corpus,a new Database and baseline for Noise-robust Audio-visual Speech Recognition

### Description
No of speakers: high-quality audio samples of 62 speakers
Total number of sentences: 6913 phonetically rich sentences
Each audio sample is sampled at 16,000 Hz
Three of the speakers are professionally-trained lipspeakers
6 types of Noises at range of SNR’s from -5db to 20 db
Babble, Cafe, Car, Living Room, White, Street

### Downloadable link for the dataset:
You can find the complete dataset here https://zenodo.org/record/260228

## Approach followed:
* Used log power spectrum of the signal as features
* Computed STFT of the signal with nfft=256, noverlap=128, nperseg=256
* STFT = log(abs(STFT))
* Trained the model with the Autoencoder decoder type network with input considering 16 frames 
* Mean Square Error loss
* Adam optimizer (default parameters)

## Frameworks:
* Keras backend

## Methods Implemented

1. Frame to frame training(Input will be noisy frame matrix, output will be clean matrix)
2. Considered the heuristic feature that the noise in the present frame depends both on the present frame and the past few frames.Based on this, trained a model considering past 7 frames and the present frame




Removing various types of noises present in the speech using Deep Neural Networks
