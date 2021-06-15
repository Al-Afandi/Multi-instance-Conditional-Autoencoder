# Multi-instance_Conditional_Autoencoder
In this repository You can find our code which impelments Multi-instance Conditional Autoencoder appraoch, descird in the paper:

*Multi-instance Conditional Autoencoder: a data driven compression model for strongly correlated datasets*
Jala Al-afandi and András Horváth

Submitted to:
The 2021 International Symposium on Broadband Multimedia Systems and Broadcasting (BMSB2021)

### Prerequisites-Installing
To run our code You need to install Python and PyTorch.

### Running our code
The repository has  many scripts implenting Multi-instance Conditional Autoencoder over two images datasets (mnist, cifar10) and two audio datasets (celebrity and Javanese). 

### Approach illustration
The figure illustrates Multi-instance autoencoder where one encoder compresses a batch of files together producing a concise latent representation. Although the three decoders receive the same data, each input can be reconstructed from its own decoder.

<img src="" width="500" height="300">

The figure illustrates multi-instance conditional autoencoder where one encoder compresses a batch of files together producing a concise latent representation. Each decoder receives the shared latent representation along with its own spacial latent variables. During training, the model is separating the shared features from the sample based features.

<img src="" width="500" height="300">

## Authors
**Jalal Al-afandi 
Andras Horvath** 
