# Anxiety prediction 

## Background
Anxiety symptoms during public health crises are associated with adverse psychiatric outcomes and impaired health decision-making. Therefore, remotely identifying the severity of short-term anxiety symptoms in the population during lockdown measures is an important public health agenda.

## Objective
This study compared two machine learning models for predicting clinical anxiety based on the Generalized Anxiety Disorder 7-item scale from time-series data of communication and social networking app usage, and anxiety-associated clinical survey variables, including cohabitation with essential workers, worries about life instability, changes in social interactions, and health status.

## Methods
The data was collected from a sample of psychiatric outpatients in Madrid, Spain, before and during the mandatory COVID-19 lockdown. Our first pipeline was based on a hidden Markov model (HMM), while in the second model, we opted for a recurrent neural network (RNN) for temporal data processing. Both architectures model the distribution of a sequence of random observations from a set of latent variables; however, in RNN, the latent variable is deterministically deduced from the current observation and the previous latent variable, while, in HMM, the set of (random) latent variables is a Markov chain. The evaluation was performed using 10-fold cross-validation due to limited data. The model accuracy and area under the receiver operating curve (AUROC) mean and standard deviation are reported.


Published in JMIR Mental Health: 
- Ryu J, Sükei E, Norbury A, H Liu S, Campaña-Montes J, Baca-Garcia E, Artés A, Perez-Rodriguez M. Shift in Social Media App Usage During COVID-19 Lockdown and Clinical Anxiety Symptoms: Machine Learning–Based Ecological Momentary Assessment Study. DOI: [10.2196/30833](https://mental.jmir.org/2021/9/e30833)
