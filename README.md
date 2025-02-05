# AI-based-identification-of-New-York-City-Open-Source-Noise-data

## Project Title: 
Urban Sound Classification

## Problem Statement:
Classify the sounds of an Urban City into 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, and street_music.

## Problem Importance:
Sounds in big urban cities can easily go over the healthy noise levels for human hearing system. For instance, it has been estimated that 9 out of 10 adults in New York City (NYC) are exposed to excessive noise levels i.e. beyond the limit of what the EPA considers to be harmful. When applied to U.S. cities of more than 4 million inhabitants, such estimates extend to over 72 million urban residents. So, it is important to identify noise sources to mitigate it.

## Dataset Information:
UrbanSound8k Dataset. It contains 27 hours of audio with 18.5 hours of annotated sound event occurrences across 10 sound classes. This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, and street_music. The classes are drawn from the urban sound taxonomy. The dataset is not preprocessed.

[UrbanSound8k Dataset Link](https://urbansounddataset.weebly.com/urbansound8k.html)

## ML Models: 
Decision Tree, Random Forest, XGBoost, Artificial Neural Network (ANN), Convolutional Neural Network (CNN), etc.

## Accuracy/Error Measures:
I would explore Accuracy Score, Precision score, Recall Score, F1-score, and F-beta score. And would prioritize the metric which seems relevant to the use case. This could also be a combination of multiple metrics.

## Previous work on the problem:
Sound analysis and classification was previously conducted by the NYU’s Music and Audio Research Lab and funded by NYU’s Centre for Urban Science
and Progress seed grant. The previous research used Weka’s classic machine learning algorithms with default parameters. The algorithms used were SVM, KNN, Decision Tree, Random Forest, and Baseline Majority vote classifier (ZeroR). My approach would be to introduce Deep Learning into the research and use an ensemble of models whether it be a combination of machine learning and deep learning models. I am new to the audio domain and would also try to find some audio preprocessing to elevate the results. If this project is successful, then it could directly add to NYU’s ongoing project SONYC.

[NYU Project SONYC](https://wp.nyu.edu/sonyc/)

## Technology used:
- PyTorch
- Tensorflow
- Librosa
