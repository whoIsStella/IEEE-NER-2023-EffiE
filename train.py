"""
    Description:
        Achieves:
            - Data Preprocessing over Ninapro DataBase5
            - Training finetune-base model (Saving weights along the way)
            - Visualize training logs (model accuracy and loss during training)
            
    Author: Jimmy L. @ SF State MIC Lab
    Date: Summer 2022
"""
import torch
from dataset import *
from model import get_model, create_finetune, train_model, plot_logs, create_dataloaders
import config


if __name__ == "__main__":
    # NOTE: Check if Utilizing GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # NOTE: Data Preprocessings
    
    # Get sEMG samples and labels. (shape: [num samples, 8(sensors/channels)])
    emg, label = folder_extract(
        config.folder_path,
        exercises=config.exercises,
        myo_pref=config.myo_pref
    )
    # Apply Standarization to data, save collected MEAN and STANDARD DEVIATION in the dataset to json
    emg = standarization(emg, config.std_mean_path)
    # Extract sEMG signals for wanted gestures.
    gest = gestures(emg, label, targets=config.targets)
    # Perform train test split
    train_gestures, test_gestures = train_test_split(gest)
    
    # NOTE: optional visualization that graphs class/gesture distributions
    # plot_distribution(train_gestures)
    # plot_distribution(test_gestures)
    
    # Convert sEMG data to signal images.
    X_train, y_train = apply_window(train_gestures, window=config.window, step=config.step)
    # Convert sEMG data to signal images.
    X_test, y_test = apply_window(test_gestures, window=config.window, step=config.step)
    
    X_train = X_train.reshape(-1, 8, config.window, 1)
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = X_test.reshape(-1, 8, config.window, 1)
    X_test = np.transpose(X_test, (0, 3, 1, 2))
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    
    print("Shape of Inputs:\n")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    print("Data Type of Inputs:\n")
    print("X_train:", X_train.dtype)
    print("_test:", X_test.dtype)
    print("\n")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=config.batch_size)
    # Get PyTorch model
    cnn = get_model(
        num_classes=config.num_classes,
        filters=config.filters,
        neurons=config.neurons,
        dropout=config.dropout,
        kernel_size=config.kernel_size,
        input_shape=config.input_shape,
        pool_size=config.pool_size
    )
    
    #Define optimizer and loss function (criterion)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=config.inital_lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Start training (And saving weights along training)
    #make sure train_model returns model, training_losses, validation_losses
    cnn, training_losses, validation_losses = train_model(
        cnn, train_loader, val_loader, optimizer, criterion,
        save_path=config.save_path, epochs=config.epochs,
        patience=config.patience, lr=config.inital_lr, device=device
    )
    
    # Visualize accuarcy and loss logs
    # plot_logs(train_acc, acc=True, save_path=config.acc_log)
    plot_logs(training_losses, validation_losses, acc=False, save_path=config.loss_log)
    
    # Load pretrained model
    model = get_model(
        num_classes=config.num_classes,
        filters=config.filters,
        neurons=config.neurons,
        dropout=config.dropout,
        kernel_size=config.kernal_size,
        input_shape=config.input_shape,
        pool_size=config.pool_size
    )
    model.load_state_dict(torch.load(config.save_path, map_location=device))
    model.to(device)
    model.eval()
    
    # # NOTE: Optional test for loaded model's performance
    # model.compile(
    #         optimizer=tf.keras.optimizers.Adam(learning_rate=0.2),
    #         loss='sparse_categorical_crossentropy',
    #         metrics=['accuracy'],
    #     )
    # # See if weights were the same
    # model.evaluate(X_test, y_test)
    
    # # # Test with finetune model. (last classifier block removed from base model)
    # # finetune_model = get_finetune(config.save_path, config.prev_params, num_classes=config.num_classes)
    # # print("finetune model loaded!")
    
    # # NOTE: You can load finetune model like this too.
    # finetune_model = create_finetune(model, num_classes=4)
    # finetune_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.2),
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy'],
    # )
    # finetune_model.evaluate(X_test, y_test)