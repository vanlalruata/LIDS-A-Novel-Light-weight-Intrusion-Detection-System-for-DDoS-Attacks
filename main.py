import os

EPOCHS = 100
BATCH_SIZE = 1024
PATH = 'H:/Datasets/CIC-DDoS2019/01-12'
n_rows = 50000

if __name__ == '__main__':


    print("1. Proposed Preprocessing: Create Dataset")
    print("2. Create PCA Dataset using processed dataset")
    print("3. Visualization")
    print("4. Proposed Model Binary Classification")
    print("5. Proposed Model Multi Classification")
    print("6. Plot Training and Validation")

    selection = int(input("Enter your implementation checker: "))

    if selection == 1:
        from Proposed.data_utils import make_datasets
        make_datasets(PATH, n_rows)

    elif selection == 2:
        from Proposed.data_utils import create_pca_dataset
        create_pca_dataset()

    elif selection == 3:
        from Proposed.data_utils import load_pca_dataset
        from vis_utils import plot_files_preprocessing, class_distributions
        from Proposed.pca_analysis import principal_component_varience, plot_pca_analysis
        print("1. Plotting Files Size Preprocessing")
        print("2. Plotting Preprocesses Class Distribution")
        print("3. Plotting Principal Component Variance")
        print("4. Plotting PCA Analysis Plot")

        dataset = load_pca_dataset()
        plot_files_preprocessing()
        class_distributions(dataset)
        plot_pca_analysis(dataset)
        principal_component_varience()

    elif selection == 4:
        from Proposed.train import trainer
        trainer(EPOCHS, BATCH_SIZE)

    elif selection == 5:
        from Proposed.train import trainer_multi
        trainer_multi(EPOCHS, BATCH_SIZE)

    elif selection == 6:
        from vis_utils import plot_proposed_model_accuracy_loss
        plot_proposed_model_accuracy_loss()
