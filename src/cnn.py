# src/cnn.py
import os
import numpy as np
from tensorflow.keras import layers, models, callbacks

def make_simple_cnn(input_shape=(64,64,1), n_classes=3):
    """
    Small CNN for phase-folded image classification.
    n_classes default 3 (confirmed/candidate/false positive)
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation='relu'))
    if n_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(layers.Dense(n_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

def train_cnn_model(model, X, y, epochs=10, batch_size=16, out_path=None):
    """
    Train CNN model on arrays X (N,H,W,C) and y (N,) integer labels.
    If out_path provided, saves model.
    """
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    model.fit(X, y, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=cb)
    if out_path:
        model.save(out_path)
    return model
