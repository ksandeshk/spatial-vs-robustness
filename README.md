# Can we have it all? On the Trade-off between Spatial and Adversarial Robustness of Neural Networks 

This repository is the codebase for the NeurIPS 2021 paper [Can we have it all? On the Trade-off between Spatial and Adversarial Robustness of Neural Networks](https://proceedings.neurips.cc/paper/2021/hash/e6ff107459d435e38b54ad4c06202c33-Abstract.html).

## Overview
The paper shows a trade-off exists between Spatial vs Robustness(Adversarial) in neural networks in a simple statistical setting. Towards achieving Pareto-optimality in this trade-off, a method (***CuSP***) is proposed based on curriculum learning that trains gradually on more difficult perturbations (both spatial and adversarial) to improve spatial and adversarial robustness simultaneously.


## Dependencies
Codebases used in the paper as is or modified accordingly.

* [GCNN] (https://github.com/adambielski/GrouPy)
* [GCNN] (https://github.com/adambielski/pytorch-gconv-experiments)
* [TRADES] (https://github.com/yaodongyu/TRADES)
* [MNIST Challenge] (https://github.com/MadryLab/mnist_challenge)
* [CIFAR10 Challenge] (https://github.com/MadryLab/cifar10_challenge)
* [advertorch] (https://github.com/BorealisAI/advertorch)


## Code documentation.

* Folder **stdcnn** : Contains the code for standard CNN(StdCNN) for VGG16 architecture
	* main_vgg16.py : Training the StdCNN/VGG16 architecture and stores the model in the current folder with sub-folder name "saved_models". Comment/Uncomment the augmentation technique to be applied in the transforms code.
	* main_vgg16_collect.py : Sample code to load and collect predictions from a pre-trained model. These predictions are used later for fooling rate calculation.
	* pgd_attack.py : PGD attack an existing trained model in the "./saved_models" folder. Uses the PGD attack code in advertorch.
	* models : Contains the various architecture models.
	* utils.py : Function to display a progress bar while training.

* Folder **gcnn** : Contains the code for group equivariant CNN(GCNN) for ResNet18 architecture (https://github.com/adambielski/pytorch-gconv-experiments)
	* train_resnet18.py : Training the GCNN/ResNet18 architecture and stores the model in the current folder with sub-folder name "saved_models". Comment/Uncomment the augmentation technique to be applied in the transforms code.
	* groupy : Contains the group operation and other code needed for GCNN architecture. (https://github.com/adambielski/GrouPy)
	* models : Contains the architecture model.
	* utils.py : Function to display a progress bar while training.
	* pgd_attack.py : PGD attack an existing trained model in the "./saved_models" folder. Uses the PGD attack code given in adversarialbox.
	* adversarialbox : Contains the files used to apply the PGD attack in the pgd_attack.py code. (advertorch based code)

* Folder **trades** : Contains code for adversarial training with PGD and TRADES and CuSP algorithm (https://github.com/yaodongyu/TRADES)
	* cusp_cifar10.py : Implementation of CuSP algorithm for CIFAR10 dataset. Comment/Uncomment the code in main() to either train with CuSP based on PGD or TRADES.
	* train_trades_cifar10.py : Comment/Uncomment the code in main() to either adversarially train with PGD/TRADES/no adversarial training.
	* models : Contains code for the various architecture models.

* Other Instructions
	* Create a "./saved_models" and "./saved_data" folder in respective folders.
	* The path to store the trained models are hard coded in the python file. Trained models are generally stored in "./saved_models" folder while other data like predictions etc. are stored in "./saved_data" folder.
	* For training with the required setting *comment/uncomment* appropriate lines in the code.
	* For plotting the fooling rate, the predictions from the network in the test code are saved and compared to the predictions with various augmentations or adversarial perturbations applied to the test data(eg. for transformations - pytorch transforms code). Eg. code in main_vgg16_collect.py and pgd_attack.py save the predictions in the "./saved_data" folder. These saved predictions are used for (standard) fooling rate calculation as given in paper. Use similar method for other architectures.

## ***Cu***rriculum based ***S***patial-Adversarial Robustness training for ***P***areto-Optimality (***CuSP***)

The algorithm performs a mix of <img src="https://render.githubusercontent.com/render/math?math=T_{\theta}(X)"> and <img src="https://render.githubusercontent.com/render/math?math=A_{\epsilon}(X)"> based on the learning rate change. Here, <img src="https://render.githubusercontent.com/render/math?math=\theta"> is the transformation budget eg. <img src="https://render.githubusercontent.com/render/math?math=\pm30^\circ"> rotation or <img src="https://render.githubusercontent.com/render/math?math=\pm 2px"> translation and <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> is the pertubation budget for adversarial perturbation eg. <img src="https://render.githubusercontent.com/render/math?math=8/255"> for CIFAR10 dataset. While this is one version of the algorithm a more general algorithm can utilize any learning rate algorithm to train the network with a combination of transformation <img src="https://render.githubusercontent.com/render/math?math=T_{\theta}(X)"> and adversarial perturbation <img src="https://render.githubusercontent.com/render/math?math=A_{\epsilon}(X)">.

<img src="https://render.githubusercontent.com/render/math?math=\phi \leftarrow AdversarialTraining_{\phi}(T_{\theta}(X) + A_{\epsilon}(T_{\theta}(X), \eta))">



## Citation

If the code related to our work is useful for your work, kindly cite this work as given below:

```[bibtex]
@inproceedings{kamath2021all,
  title={Can we have it all? On the Trade-off between Spatial and Adversarial Robustness of Neural Networks}, 
  author={Sandesh Kamath and Amit Deshpande and K V Subrahmanyam and Vineeth N Balasubramanian},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  howpublished={arXiv preprint arXiv:2002.11318},
  url={https://openreview.net/forum?id=k9iBo3RmCFd}
}

```
