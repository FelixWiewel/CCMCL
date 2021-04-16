#!/usr/bin/python3.6
"""
Script for training
"""

import tensorflow as tf
import models
import datasets
import utils
import numpy as np

utils.enable_gpu_mem_growth()

# Define constants
BATCH_SIZE = 128
ITERS = 1000
VAL_ITERS = 10
VAL_BATCHES = 10
LEARNING_RATE = 0.01
TASKS = 5
CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
BUFFER_SIZES = (100, 200, 300, 400, 500)
DIST_BATCH_SIZE = 256
DIST_LEARNING_RATE = 0.01
IMG_SHAPE = (32, 32, 3)
K = 100
T = 10
RUNS = 5
LOG_PATH = "../logs/CompositionalBuffer/SVHN"

# Create array for storing results
res_loss = np.zeros((RUNS, len(BUFFER_SIZES)), dtype=np.float)
res_acc = np.zeros((RUNS, len(BUFFER_SIZES)), dtype=np.float)

for i, BUFFER_SIZE in enumerate(BUFFER_SIZES):
    for run in range(RUNS):
        # Instantiate model and trainer
        model = models.CNN(10)
        model.build((None, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
        buf = models.CompositionalBalancedBuffer()
        train = utils.Trainer()

        # Instantiate optimizer
        optimizer = tf.keras.optimizers.SGD(LEARNING_RATE)

        # Load data set
        ds = datasets.SplitSVHN(num_validation=VAL_BATCHES*BATCH_SIZE)
        _, val_ds, test_ds = ds.get_all()
        val_ds = val_ds.cache().repeat().batch(BATCH_SIZE).map(utils.standardize)
        test_ds = test_ds.cache().batch(BATCH_SIZE).map(utils.standardize)

        # Train on sequence
        for t in range(TASKS):
            # Determine classes
            classes = CLASSES[2 * t:2 * (t + 1)]
            # Load training data set
            train_ds, _, _ = ds.get_split(classes)
            train_ds = train_ds.cache().repeat().shuffle(10000).batch(BATCH_SIZE).map(utils.standardize)
            # Distill
            print("Distilling classes {}..".format(classes))
            # Compress and store
            for c in classes:
                # Load data set
                train_ds, _, _ = ds.get_split(c)
                train_ds = train_ds.cache().repeat().shuffle(10000).batch(DIST_BATCH_SIZE).map(utils.standardize)
                buf.compress_add(train_ds, c, DIST_BATCH_SIZE, LEARNING_RATE, DIST_LEARNING_RATE, IMG_SHAPE,
                                 int(BUFFER_SIZE / len(CLASSES)), K, T, model)
            buf.summary()
            # Training loop
            m_train_loss = tf.keras.metrics.Mean()
            m_val_acc = tf.keras.metrics.Accuracy()
            m_val_loss = tf.keras.metrics.Mean()
            utils.reinitialize_model(model)
            for iters in range(ITERS):
                # Sample a batch from the buffer and train
                x_r, y_r = buf.sample(BATCH_SIZE)
                current_loss = train.train_step(x_r, y_r, model, optimizer)
                m_train_loss.update_state(current_loss)
                if iters % VAL_ITERS == 0:
                    # Validation
                    val_iters = 0
                    for x, y in val_ds:
                        val_iters += 1
                        logits = model(x, training=False)
                        current_loss = tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True)
                        m_val_loss.update_state(current_loss)
                        m_val_acc.update_state(tf.argmax(y, axis=-1), tf.argmax(logits, axis=-1))
                        if val_iters == VAL_BATCHES:
                            # Get metrics
                            train_loss = m_train_loss.result()
                            m_train_loss.reset_states()
                            val_acc = m_val_acc.result()
                            m_val_acc.reset_states()
                            val_loss = m_val_loss.result()
                            m_val_loss.reset_states()
                            print("Task: {} Iter: {} Train Loss: {:.3} Val Loss: {:.3} Val Accuracy: {:.3}".format(t,
                                                                                                                   iters,
                                                                                                                   train_loss,
                                                                                                                   val_loss,
                                                                                                                   val_acc))
                            # Reset validation iterations
                            val_iters = 0
                            break

        # Test model on complete data set
        m_test_acc = tf.keras.metrics.Accuracy()
        m_test_loss = tf.keras.metrics.Mean()
        for x, y in test_ds:
            logits = model(x, training=False)
            current_loss = tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True)
            m_test_acc.update_state(tf.argmax(y, axis=-1), tf.argmax(logits, axis=-1))
            m_test_loss.update_state(current_loss)
        test_loss = m_test_loss.result()
        test_acc = m_test_acc.result()
        print("Test Loss: {:.3} Test Accuracy {:.3}".format(test_loss, test_acc))
        # Store result
        res_loss[run, i] = test_loss
        res_acc[run, i] = test_acc

        # Write results
        print("Saving results to {}...".format(LOG_PATH))
        np.save(LOG_PATH + "/acc.npy", res_acc)
        np.savetxt(LOG_PATH + "/acc.log", res_acc, fmt="%.4f", delimiter=";",
                   header="Buffer sizes: " + str(BUFFER_SIZES))
        np.save(LOG_PATH + "/loss.npy", res_loss)
        np.savetxt(LOG_PATH + "/loss.log", res_loss, fmt="%.4f", delimiter=";",
                   header="Buffer sizes: " + str(BUFFER_SIZES))
        np.save(LOG_PATH + "/w_" + str(BUFFER_SIZE) + ".npy", buf.w_buffer)
        np.save(LOG_PATH + "/c_" + str(BUFFER_SIZE) + ".npy", buf.c_buffer)
