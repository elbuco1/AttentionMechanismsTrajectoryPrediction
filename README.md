# Study of attention mechanisms for trajectory prediction in Deep Learning
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

A whole lot of models have been proposed to use those informations for trajectory prediction. Amongst them, some tried to use attention mechanisms. Attention mechanisms first came from Natural Language Processing (NLP) and were then used in Computer Vision (CV). 


The purpose of such mechanisms is to select automatically, based on the prediction context, which elements from a set of observations are relevant for the current prediction. For instance in NLP, one main task is language translation, which consists in given an input sentence in a language, to output its translation in another language. In this context, the set of elements is made of the words of the input sentence. At prediction time, the words from the output sentence are predicted sequentially. Attention mechanisms come from the observation that for a given predicted word, not every word in the input sentence is relevant. Therefore, attention mechanisms can be used to select which input words are relevant for every predicted word, making it possible to modify the input based on context.

In trajectory prediction, attention mechanisms are used for taking into account two things: on one hand for social interactions, on the other hand for spatial interactions. In the case of social interactions, attention mechanisms are used to select which agents must be considered from the surrounding of the agent we want to predict the future position (or main agent). In the case of spatial interactions, attention mechanisms are used to select which physical part of the scene (based on a top-view image) might have an impact on the future trajectory of the agent.



### Goals of the project

The main goal of this project was to evaluate the attention mechanisms already used for trajectory prediction. The attention mechanism were transposed somewhat "naïvely" from NLP to trajectory prediction. We modify those models and try to reduce drastically the computing time while keeping the same prediction quality, showing that such transposition was indeed naïve.

While addressing this goal, we started to question the evaluation settings and in particular the relevance of the metrics used to evaluate such models. We therefore try to propose a new set of metrics to enrich the comparison of models.

Finally despite the claims of previous studies, that such models could take into account social and spatial interactions we try to show that it might not be the case (in the continuity of https://arxiv.org/abs/1903.07933?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29).


## Study
### Baselines
#### Soft-attention
Previous studies using attention mechanisms for trajectory prediction are based on its soft-attention variant. The soft-attention can be seen as a function A(.,.) of two inputs Q and V_i. Q and V_i are attribute vectors. The function A(Q, V_i) is a feed-forward neural network  whose output is of size 1. Basically, Q and V_i are concatenated and fed in the neural network which outputs a positive real number. In the function, Q embodies the prediction context and V_i is an element for which we want to evaluate the relevance for the next prediction. The ouput S_i of A(Q, V_i) is to be interpreted as the relevance of V_i for the next prediction based on the current prediction context Q. This operation is repeated for every V_i from a set V. The scores are then normalized using a softmax function so that they sum to one. This normalization results in the attention weights W. The attention vector is the weighted sum of attribute vectors V_i by its attention weight W_i. The intuition behind it is that the attention vector contains informations from elements V_i proportionally to their relevance for the coming prediction. This process is differentiable and therefore easy to use with backpropagation in deep learning.

[insérer image]

### The baseline for attentive trajectory prediction

In the litterature, all the models for attentive trajectory prediction are based upon the sequence-to-sequence architecture. First the observed trajectory of the agent we want to predict the next position is fed into an LSTM (the encoder) that extracts automatically a fixed-size attribute vector. This vector is used to initialize the hidden state of another LSTM (the decoder) which will be used to predict the following positions. The hidden state of the decoder represents the prediction context. It is first initialized based on the observed trajectory and then updated gradually as we go along the prediction process. The main difference between the "naïve" sequence-to-sequence model and the attentive one takes place during the prediction part. Before, we were taking as decoder input either the last position of the observed trajectory or the last predicted position and so on recursively. In the attentive alternative, for each prediction, we call the soft-attention module and concatenate the attention vector to the decoder regular input. 

[insérer image] 

One thing to notice is that the module is called for every prediction. This comes from the natural language processing use of this model, for instance for language prediction where it is apropos to select the relevant words in the input sentence for every predicted word.
Another thing is that only the first parameter of the attention module changes from one call to another. It is the parameter Q from the previous paragraph, namely the prediction context. The value of the prediction context is the current decoder hidden state. The second parameter, namely the set of element whose relevance is to be evaluated, doesn't change from one call to another.

The only thing that changes between the attentive model for social interactions (social attention) and the attentive model for spatial interactions (spatial attention) is the set of elements V.

#### Set of element for social attention
In social attention we want to give a weight to every pair of (main agent, ith neighbor), main agent being the agent we want to predict the upcoming path and the neighbors being the other agents in the scene during observation time. To that end, every V_i in that context corresponds to the extracted attribute vector (using the same LSTM encoder as before) from the observed trajectory of the ith neighbor.

[insert image]

#### Set of element for spatial attention

In spatial attention we want to give a weight to every pair of (main agent, ith scene part), main agent being the agent we want to predict the upcoming path. To get a spatial representation of the scene, we have a top-view image of the scene (always the same image). We pass it through a CNN, VGG-19 pre-trained on the image-net dataset, resulting in some feature maps. Each feature map corresponds to an attribute vector caracterizing a part of the image (grid-like division of the image). Those vectors are the elements in the set V.

[insert image]


#### Relevance of naïve transposition from NLP

We claim that such a transposition of attention mechanisms from NLP to trajectory prediction is somewhat a naïve one, in that it used very similarly as in NLP without giving much thinking on its relevance for trajectory prediction. Our main observation is based upon the fact that these models compute, for each prediction an attention vector. Yet the set of elements to be considered in the soft-attention module stays the same between every prediction. The only parameter that changes is the context vector given by the hidden state of the decoder. In addition, since we predict a position every 0.4s, it might not be relevant to reassess at this rate the interactions between the main agent and its environment. We want to show that computing the attention vector just once in order to predict the whole future trajectory gives similar results.

### The proposed model to test the claim

The model we propose to back up our claim is very simple. First, we use a CNN to extract useful information from the observed trajectory of the main agent as a feature vector. We are using a CNN since it was shown that it gives better results than LSTM. We then call the soft-attention module on our set of elements using this attribute vector as context. We then concatenate the resulting attention vector and the attribute vector together and feed it into a Feed-Forward Neural Network that predicts simultaneously all the future positions. We make a simultaneous prediction since it has been shown that it allows to get rid of the error accumulation issue of recursive prediction using an LSTM.
