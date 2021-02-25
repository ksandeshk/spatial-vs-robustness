Folder gcnn : Contains the code for group equivariant CNN(GCNN) for ResNet18 architecture for CIFAR-10 dataset
train_resnet18.py : Training the GCNN/ResNet18 architecture and stores the model in the current folder with sub-folder name "saved_models". Comment/Uncomment the augmentation technique to be applied in the transforms code.
groupy : This folder contains the group operation and other code needed for GCNN architecture.
models : This folder contains the architecture model.
utils.py : Has function to display a progress bar while training.
pgd_attack.py : PGD attack an existing trained model in the "./saved_models" folder. Uses the PGD attack code given in adversarialbox.
adversarialbox : This folder contains the files used to apply the PGD attack in the pgd_attack.py code.

For plotting the fooling rate, the predictions from the network in the test code are saved and compared to the predictions with various augmentations or adversarial perturbations applied to the test data(eg. for transformations - pytorch transforms code). Eg. code in main_vgg16_collect.py and pgd_attack.py save the predictions in the "./saved_data" folder. These saved predictions are used for (standard) fooling rate calculation as given in paper. Use similar method for other architectures.

