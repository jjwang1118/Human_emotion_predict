以下為模型連結:
1. my_model:https://drive.google.com/drive/folders/1pbsPBVw7OyDqaj3HxbJcKkdSmoES5Rwv?usp=drive_link
2. my_model_01:https://drive.google.com/drive/folders/1pbsPBVw7OyDqaj3HxbJcKkdSmoES5Rwv?usp=drive_link
2. my_model_02:https://drive.google.com/drive/folders/1pbsPBVw7OyDqaj3HxbJcKkdSmoES5Rwv?usp=drive_link
3. my_model_03:https://drive.google.com/drive/folders/1pbsPBVw7OyDqaj3HxbJcKkdSmoES5Rwv?usp=drive_link
--------------------------------------------------------------------------------------------------

參數如下:
1.my_model:
    num_epochs = 100
    batch_size = 50
    num_features = 64
    rate_drop = 0.1
    accuracy: 0.6014 - loss: 22.1459 
    val_accuracy: 0.6002 - val_loss: 22.1019
 --------------------------------------------------------------------------------------------------
2.my_model_01:
    num_epochs = 200
    batch_size = 70
    num_features = 64
    rate_drop = 0.1
    accuracy: 0.5974 - loss: 35.1169 
    val_accuracy: 0.5912 - val_loss: 35.1022
--------------------------------------------------------------------------------------------------
3.my_model_02:
    num_epochs = 200
    batch_size = 128
    num_features = 64
    rate_drop = 0.1
    accuracy: 0.5199 - loss: 63.3161 
    val_accuracy: 0.5408 - val_loss: 63.2663 
--------------------------------------------------------------------------------------------------
4.my_model_03:
    num_epochs = 100
    batch_size = 64
    num_features = 64
    rate_drop = 0.3
    accuracy: 0.5866 - loss: 30.9005 
    val_accuracy: 0.5765 - val_loss: 30.8504 
--------------------------------------------------------------------------------------------------
5.my_model_04:
    加一層
    model.add(Conv2D(512, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))    
    1accuracy: 0.5541 - loss: 57.7011
    val_accuracy: 0.5433 - val_loss: 57.6390 
--------------------------------------------------------------------------------------------------
6.my_model_05:
    4028→2048
    model.add(Dense(2048, activation="relu", kernel_regularizer=l2()))
    model.add(Conv2D(256, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
--------------------------------------------------------------------------------------------------
