Folder trades : Contains code for adversarial training with PGD and TRADES for CIFAR-10 dataset
train_trades_cifar10.py : Comment/Uncomment the code in main() to either adversarially train with PGD/TRADES/no adversarial training.
models : This folder contains the various architecture models.

For plotting the fooling rate, the predictions from the network in the test code are saved and compared to the predictions with various augmentations or adversarial perturbations applied to the test data(eg. for transformations - pytorch transforms code). Eg. code in main_vgg16_collect.py and pgd_attack.py save the predictions in the "./saved_data" folder. These saved predictions are used for (standard) fooling rate calculation as given in paper. Use similar method for other architectures.
