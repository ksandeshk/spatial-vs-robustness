Folder stdcnn : Contains the code for standard CNN(StdCNN) for VGG16 architecture for CIFAR-10 dataset
main_vgg16.py : Training the StdCNN/VGG16 architecture and stores the model in the current folder with sub-folder name "saved_models". Comment/Uncomment the augmentation technique to be applied in the transforms code.
main_vgg16_collect.py : Sample code to load and collect predictions from a pre-trained model. These predictions are used later for fooling rate calculation.
pgd_attack.py : PGD attack an existing trained model in the "./saved_models" folder. Uses the PGD attack code in advertorch.
models : This folder contains the various architecture models.
utils.py : Has function to display a progress bar while training.

For plotting the fooling rate, the predictions from the network in the test code are saved and compared to the predictions with various augmentations or adversarial perturbations applied to the test data(eg. for transformations - pytorch transforms code). Eg. code in main_vgg16_collect.py and pgd_attack.py save the predictions in the "./saved_data" folder. These saved predictions are used for (standard) fooling rate calculation as given in paper. Use similar method for other architectures.

