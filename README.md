# Study of attention mechanisms for trajectory prediction
## Overview

### Problem definition
This project addresses the task of predicting future trajectories of interacting agents in a scene. We refer to this problem as trajectory prediction. The trajectory prediction task works as follows: given the past observed trajectory of an agent (pedestrian, cyclist, driver ...) we want to predict the future trajectory of the agent, i.e. the ordered next positions it will cross in the future. In the litterature, the prediction task is commonly tackled for 8 observed positions and 12 predicted positions. The time between two positions being 0.4s, we observe an agent moving during 3.2s and predict its movement for the next 4.8s.

#### Naïve prediction
We refer to this task as naïve prediction when the only information used to predict an agent's future path is its past observed trajectory. This naïve prediction is most of the time tackled using deep learning models such as LSTM-MLP or sequence-to-sequence LSTM. In the sequence-to-sequence LSTM model, the first LSTM (encoder) takes as input the observed trajectory and extracts recursively a fixed-size representation of it. Using this representation, the second LSTM (decoder) predicts recursively the future positions of the agent. By recursively, we mean that it first predicts the next position and then uses it to make the following prediction and so on.
It has been shown that predicting recursively the future trajectory leads to an error accumulation along the predicted position. Indeed since the model bases its prediction on previously made predictions, it depends on the error previously made on those. To address this issue, one can use the LSTM-MLP model that replaces de LSTM decoder with a simple Multi Layer Perceptron( MLP) network. This simpler model predicts all the future positions simultaneously getting rid of the error accumulation issue.

Finally, it's been shown that against all expectations, using a Convolutionnal Neural Network (CNN) along the temporal dimension instead of Recurrent Neural Network (RNN) gives slightly better results when predicting the future trajectory and is way faster due to the fact that convolution operations can be made in parallel whereas operations in an RNN are made sequentially. We refer in this project to this model as CNN-MLP.

All three models are implemented in this project and referred as seq2seq, rnn-mlp and cnn-mlp.

#### Using the environment
One main research question in this field, is to train a model on a set of scenes and test it on a new set of scenes. In other words, we want a model capable of generalizing its learning across different and unseen environments.

It is obvious that such a challenge can't be overcome using only past observed trajectory as input for our models. In fact, when an agent crosses a scene, the scene has an influence on its motion. The agent interacts with the scene. Mainly two types of interactions are addressed in the litterature:
Social interactions refer to the influence the interactions between an agent and its surrounding agents have on their motions. 

Spatial interactions refer to the influence between an agent and the physical constraints of a scene (such as roads, trees, obstacles ...) on the agent motion.

A model good enough to generalize knowledge to a new set of scenes should be able to make good use of those interactions.


attention mechanisms 
NLP 
Computer vision

### Goals of the project

The main goal of this project was first to evaluate the attention mechanisms already used
