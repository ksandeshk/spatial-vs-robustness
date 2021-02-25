# spatial-vs-robustness

Preliminary version of the code used to obtain results for the paper https://arxiv.org/abs/2002.11318. The code uses a copy from the original source and added methods to obtain information needed for the paper. Kindly refer to the paper for more details.

1) Folder stdcnn : Contains the code for standard CNN(StdCNN) for VGG16 architecture for CIFAR-10 dataset
main_vgg16.py : Training the StdCNN/VGG16 architecture and stores the model in the current folder with sub-folder name "saved_models". Comment/Uncomment the augmentation technique to be applied in the transforms code.
main_vgg16_collect.py : Sample code to load and collect predictions from a pre-trained model. These predictions are used later for fooling rate calculation.
pgd_attack.py : PGD attack an existing trained model in the "./saved_models" folder. Uses the PGD attack code in advertorch.
models : This folder contains the various architecture models.
utils.py : Has function to display a progress bar while training.

2) Folder gcnn : Contains the code for group equivariant CNN(GCNN) for ResNet18 architecture for CIFAR-10 dataset
train_resnet18.py : Training the GCNN/ResNet18 architecture and stores the model in the current folder with sub-folder name "saved_models". Comment/Uncomment the augmentation technique to be applied in the transforms code.
groupy : This folder contains the group operation and other code needed for GCNN architecture.
models : This folder contains the architecture model.
utils.py : Has function to display a progress bar while training.
pgd_attack.py : PGD attack an existing trained model in the "./saved_models" folder. Uses the PGD attack code given in adversarialbox.
adversarialbox : This folder contains the files used to apply the PGD attack in the pgd_attack.py code.

3) Folder trades : Contains code for adversarial training with PGD and TRADES for CIFAR-10 dataset
train_trades_cifar10.py : Comment/Uncomment the code in main() to either adversarially train with PGD/TRADES/no adversarial training.
models : This folder contains the various architecture models.

For plotting the fooling rate, the predictions from the network in the test code are saved and compared to the predictions with various augmentations or adversarial perturbations applied to the test data(eg. for transformations - pytorch transforms code). Eg. code in main_vgg16_collect.py and pgd_attack.py save the predictions in the "./saved_data" folder. These saved predictions are used for (standard) fooling rate calculation as given in paper. Use similar method for other architectures.
