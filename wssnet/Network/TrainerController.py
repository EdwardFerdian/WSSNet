"""
Flat WSS Network - TrainerController
Author: Edward Ferdian
Date:   27/01/2021
"""

import tensorflow as tf
import numpy as np
import datetime
import time
import shutil
import os
import pickle
from wssnet.Network.WSSNet import WSSNet
from wssnet.Network import utility, h5util
from icecream import ic
import config
import math

class TrainerController:
    # constructor
    def __init__(self, input_shape, use_vector, initial_learning_rate=1e-4, lr_decay=None, quicksave_enable=True, network_name='FlatMapWSS', use_tangential_velocity=False):
        """
            TrainerController constructor
            Setup all the placeholders, network graph, loss functions and optimizer here.
        """
        # General param
        # Training params
        self.QUICKSAVE_ENABLED = quicksave_enable
        self.use_vector = use_vector
        
        # Network
        self.network_name = network_name

        self.lr_decay  = lr_decay
        self.l2_weight = 1e-2

        self.reset_epoch = 10
        self.lr_max = 5e-4
        self.lr_min = 5e-7
        
        # Prepare Input 
        xyz0 = tf.keras.layers.Input(shape=input_shape + (3,), name='xyz0')
        xyz1 = tf.keras.layers.Input(shape=input_shape + (3,), name='xyz1')
        xyz2 = tf.keras.layers.Input(shape=input_shape + (3,), name='xyz2')

        if use_tangential_velocity:
            v1 = tf.keras.layers.Input(shape=input_shape + (1,), name='v1')
            v2 = tf.keras.layers.Input(shape=input_shape + (1,), name='v2')
        else:
            v1 = tf.keras.layers.Input(shape=input_shape + (3,), name='v1')
            v2 = tf.keras.layers.Input(shape=input_shape + (3,), name='v2')

        # prep input layer
        input_layer = [xyz0, xyz1, xyz2, v1, v2]

        # build the network        
        net = WSSNet()
        self.predictions = net.build_network(input_layer, use_vector)
        self.model = tf.keras.Model(input_layer, self.predictions)

        # ===== Loss function =====
        if lr_decay == 'cosine':
            self.learning_rate = self.lr_max
        else:
            self.learning_rate = initial_learning_rate
            self.initial_learning_rate = initial_learning_rate
        
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        
        # Metric dictionary
        self.loss_metrics = dict([
            ('train_loss', tf.keras.metrics.Mean(name='train_loss')),
            ('val_loss', tf.keras.metrics.Mean(name='val_loss')),

            ('train_mse', tf.keras.metrics.Mean(name='train_mse')),
            ('val_mse', tf.keras.metrics.Mean(name='val_mse')),

            ('train_ssim', tf.keras.metrics.Mean(name='train_ssim')),
            ('val_ssim', tf.keras.metrics.Mean(name='val_ssim')),

            ('L2_reg', tf.keras.metrics.Mean(name='L2_reg')),

            # ('train_accuracy', tf.keras.metrics.Mean(name='train_accuracy')),
            # ('val_accuracy', tf.keras.metrics.Mean(name='val_accuracy')),
        ])
        self.accuracy_metric = 'val_loss'

    def init_model_dir(self):
        """
            Create model directory to save the weights with a [network_name]_[datetime] format
            Also prepare logfile and tensorboard summary within the directory.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.unique_model_name = f'{self.network_name}_{timestamp}'

        self.model_dir = f"{config.MODEL_DIR}/{self.unique_model_name}"
        # Do not use .ckpt on the model_path
        self.model_path = f"{self.model_dir}/{self.network_name}"

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        # summary - Tensorboard stuff
        self._prepare_logfile_and_summary()
    
    def _prepare_logfile_and_summary(self):
        """
            Prepare csv logfile to keep track of the loss and Tensorboard summaries
        """
        # summary - Tensorboard stuff
        self.train_writer = tf.summary.create_file_writer(self.model_dir+'/tensorboard/train')
        self.val_writer = tf.summary.create_file_writer(self.model_dir+'/tensorboard/validate')

        # Prepare log file
        self.logfile = self.model_dir + '/loss.csv'

        utility.log_to_file(self.logfile, f'Network: {self.network_name}\n')
        utility.log_to_file(self.logfile, f'Initial learning rate: {self.learning_rate}\n')
        utility.log_to_file(self.logfile, f'Accuracy metric: {self.accuracy_metric}\n')
        utility.log_to_file(self.logfile, f'Lr_decay: {self.lr_decay}\n')
        utility.log_to_file(self.logfile, f'L2_reg: {self.l2_weight}\n')
        
        # Header
        stat_names = ','.join(self.loss_metrics.keys()) # train and val stat names
        utility.log_to_file(self.logfile, f'epoch, {stat_names}, learning rate, elapsed (sec), best_model, benchmark_err\n')
        

        print("Copying source code to model directory...")
        # Copy all the source file to the model dir for backup
        directory_to_backup = [".", "Network", "utility"]
        for directory in directory_to_backup:
            dirpath = f"{config.CODE_DIR}/{directory}"
            files = os.listdir(dirpath)
            for fname in files:
                if fname.endswith(".py") or fname.endswith(".ipynb"):
                    dest_fpath = os.path.join(self.model_dir,"backup_source",directory, fname)
                    os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)

                    shutil.copy2(f"{dirpath}/{fname}", dest_fpath)

    def calculate_ssim(self, wss_true, wss_pred):
        max_vals = tf.reduce_max(wss_true, axis=(1,2,3))
        # tf ssim only receive 1 global max_val, even if we provide multiple ones
        # so we normalize the wss first to its own max_val then calculate ssim
        wss_pred_norm = wss_pred / max_vals[:,tf.newaxis,tf.newaxis,tf.newaxis]
        wss_true_norm = wss_true / max_vals[:,tf.newaxis,tf.newaxis,tf.newaxis]

        ssim_tf = tf.image.ssim(wss_pred_norm, wss_true_norm, max_val=1)
        return ssim_tf

    def loss_function(self, data_pairs, prediction_pairs):
        pred_wss = prediction_pairs
        true_wss = data_pairs['wss']

        # apply mask
        wss_mask = data_pairs['mask']
        pred_wss = pred_wss * wss_mask
        true_wss = true_wss * wss_mask
        
        wss_diff = pred_wss - true_wss
        mae = tf.abs(wss_diff)

        if self.use_vector:
            eps = 1e-12
            # if we don't add epsilon, sqrt can produce nan
            true_wss = tf.sqrt(tf.reduce_sum(true_wss ** 2, axis=-1, keepdims=True) + eps)
            pred_wss = tf.sqrt(tf.reduce_sum(pred_wss ** 2, axis=-1, keepdims=True) + eps)

            ssim = self.calculate_ssim(true_wss, pred_wss)
        else:    
            ssim = self.calculate_ssim(true_wss, pred_wss)

        loss = tf.reduce_mean(mae) +  1.5 * (1-ssim)
        return loss, mae, ssim
        

    def calculate_and_update_metrics(self, data_pairs, prediction_pairs, metric_set):
        loss, mse, ssim = self.loss_function(data_pairs, prediction_pairs)

        if metric_set == 'train':
            # --- Regularization --- 
            l2_reg = self.get_regularization_loss()
            m = len(data_pairs['xyz0'])
            l2_reg = l2_reg * (self.l2_weight / (2 * m))

            self.loss_metrics[f'L2_reg'].update_state(l2_reg)

            loss = tf.reduce_mean(loss) + l2_reg
            # --- end of regularization ---

        # Update the loss and accuracy
        self.loss_metrics[f'{metric_set}_loss'].update_state(loss)
        self.loss_metrics[f'{metric_set}_mse'].update_state(mse)
        self.loss_metrics[f'{metric_set}_ssim'].update_state(ssim)

        return loss
        
    def get_regularization_loss(self):
        """
            https://stackoverflow.com/questions/38286717/tensorflow-regularization-with-l2-loss-how-to-apply-to-all-weights-not-just
        """
        lossL2 = [ tf.reduce_sum(v**2) for v in self.model.trainable_variables
                    if 'kernel' in v.name ]
        return tf.reduce_sum(lossL2)

    @tf.function
    def train_step(self, data_pairs):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            input_data = [data_pairs['xyz0'],data_pairs['xyz1'], data_pairs['xyz2'], data_pairs['v1'], data_pairs['v2']]
            # input_data = tf.keras.layers.concatenate(input_data)
            wss_pred = self.model(input_data, training=True)

            loss = self.calculate_and_update_metrics(data_pairs, wss_pred, 'train')
                   
        # Get the gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Update the weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    @tf.function
    def test_step(self, data_pairs):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        input_data = [data_pairs['xyz0'],data_pairs['xyz1'], data_pairs['xyz2'], data_pairs['v1'], data_pairs['v2']]
        wss_pred = self.model(input_data, training=False)

        self.calculate_and_update_metrics(data_pairs, wss_pred, 'val')

    def quicksave(self, testset, epoch_nr):
        """
            Predict a batch of data from the benchmark testset.
            This is saved under the model directory with the name quicksave_[network_name].h5
            Quicksave is done everytime the best model is saved.
        """
        for i, (data_pairs) in enumerate(testset):

            input_data = [data_pairs['xyz0'],data_pairs['xyz1'], data_pairs['xyz2'], data_pairs['v1'], data_pairs['v2']]
            wss_pred = self.model.predict(input_data)

            loss, mse, ssim = self.loss_function(data_pairs, wss_pred)
            
            # ------ Saving ------
            quicksave_filename = f"quicksave_{self.network_name}.h5"
            h5util.save_predictions(self.model_dir, quicksave_filename, "epoch", np.asarray([epoch_nr]), compression='gzip')

            # Expand dim to [epoch_nr, batch, ....]
            wss_pred = np.expand_dims(wss_pred, 0)
            h5util.save_predictions(self.model_dir, quicksave_filename, f"wss", wss_pred, compression='gzip')

            if epoch_nr == 1:
                # Save the actual data only for the first epoch
                h5util.save_predictions(self.model_dir, quicksave_filename, "xyz0", data_pairs[f'xyz0'], compression='gzip')
                h5util.save_predictions(self.model_dir, quicksave_filename, "xyz1", data_pairs[f'xyz1'], compression='gzip')
                h5util.save_predictions(self.model_dir, quicksave_filename, "xyz2", data_pairs[f'xyz2'], compression='gzip')

                h5util.save_predictions(self.model_dir, quicksave_filename, "v1", data_pairs[f'v1'], compression='gzip')
                h5util.save_predictions(self.model_dir, quicksave_filename, "v2", data_pairs[f'v2'], compression='gzip')

                h5util.save_predictions(self.model_dir, quicksave_filename, "_idx", data_pairs[f'idx'], compression='gzip')

                
                h5util.save_predictions(self.model_dir, quicksave_filename, f"wss_true", data_pairs[f'wss'], compression='gzip')
        
            # Do only 1 batch
            break
        return loss

    def cosine_learning_rate(self, curr_epoch):
        curr_epoch = curr_epoch % self.reset_epoch
        new_lr = self.lr_min + 0.5*(self.lr_max-self.lr_min) * (1 + tf.math.cos(math.pi*curr_epoch/self.reset_epoch))
        self.optimizer.lr = new_lr

    def decay_learning_rate(self, curr_epoch):
        self.optimizer.lr = self.initial_learning_rate / (curr_epoch + 1) ** 0.5

    def reset_metrics(self):
        for key in self.loss_metrics.keys():
            self.loss_metrics[key].reset_states()

    def train_network(self, trainset, valset, n_epoch, testset=None):
        """
            Main training function. Receives trainining and validation TF dataset.
        """
        # ----- Run the training -----
        print("==================== TRAINING =================")
        print(f'Learning rate {self.optimizer.lr.numpy():.7f}')
        print(f"Start training at {time.ctime()} - {self.unique_model_name}\n")
        start_time = time.time()
        
        # Setup acc and data count
        previous_loss = np.inf
        total_batch_train = tf.data.experimental.cardinality(trainset).numpy()
        total_batch_val = tf.data.experimental.cardinality(valset).numpy()

        for epoch in range(n_epoch):
            # ------------------------------- Training -------------------------------
            if self.lr_decay == 'cosine':
                self.cosine_learning_rate(epoch)
            elif self.lr_decay == 'decay':
                self.decay_learning_rate(epoch)

            # Reset the metrics at the start of the next epoch
            self.reset_metrics()

            start_loop = time.time()
            # --- Training ---
            for i, (data_pairs) in enumerate(trainset):
                # Train the network
                self.train_step(data_pairs)
                message = f"Epoch {epoch+1} Train batch {i+1}/{total_batch_train} | loss: {self.loss_metrics['train_loss'].result():.5f} - {time.time()-start_loop:.1f} secs"
                print(f"\r{message}", end='')

            # --- Validation ---
            for i, (data_pairs) in enumerate(valset):
                self.test_step(data_pairs)
                message = f"Epoch {epoch+1} Validation batch {i+1}/{total_batch_val} | loss: {self.loss_metrics['val_loss'].result():.5f} - {time.time()-start_loop:.1f} secs"
                print(f"\r{message}", end='')

            # --- Epoch logging ---
            message = f"\rEpoch {epoch+1} Train loss: {self.loss_metrics['train_loss'].result():.5f}, Val loss: {self.loss_metrics['val_loss'].result():.5f} - {time.time()-start_loop:.1f} secs"
            
            loss_values = []
            # Get the loss values from the loss_metrics dict
            for key, value in self.loss_metrics.items():
                # TODO: handle formatting here
                loss_values.append(f'{value.result():.7f}')
            loss_str = ','.join(loss_values)
            log_line = f"{epoch+1},{loss_str},{self.optimizer.lr.numpy():.6f},{time.time()-start_loop:.1f}"
            
            self._update_summary_logging(epoch)

            # --- Save criteria ---
            if self.loss_metrics[self.accuracy_metric].result() < previous_loss:
                
                self.save_best_model()
                
                # Update best acc
                previous_loss = self.loss_metrics[self.accuracy_metric].result()
                
                # logging
                message  += ' **' # Mark as saved
                log_line += ',**'

                # Benchmarking
                if self.QUICKSAVE_ENABLED and testset is not None:
                    quick_loss = self.quicksave(testset, epoch+1)
                    quick_loss = np.mean(quick_loss)
                    message  += f' Benchmark loss: {quick_loss:.5f}'
                    log_line += f', {quick_loss:.7f}'
            # Logging
            print(message)
            utility.log_to_file(self.logfile, log_line+"\n")

            # /END of epoch loop

        # End
        hrs, mins, secs = utility.calculate_time_elapsed(start_time)
        message =  f"\nTraining {self.network_name} completed! - name: {self.unique_model_name}"
        message += f"\nTotal training time: {hrs} hrs {mins} mins {secs} secs."
        message += f"\nFinished at {time.ctime()}"
        message += f"\n==================== END TRAINING ================="
        utility.log_to_file(self.logfile, message)
        print(message)
        # Finish!

    def save_best_model(self):
        """
            Save model weights and also optmizer weights to enable restore model
            to continue training

            Based on:
            https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
        """
        # Save model weights.
        self.model.save(f'{self.model_path}-best.h5')
        
        # Save optimizer weights.
        symbolic_weights = getattr(self.optimizer, 'weights')
        if symbolic_weights:
            weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
            with open(f'{self.model_dir}/optimizer.pkl', 'wb') as f:
                pickle.dump(weight_values, f)

    def restore_model(self, old_model_dir, old_model_file):
        """
            Restore model weights and optimizer weights for uncompiled model
            Based on: https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state

            For an uncompiled model, we cannot just set the optmizer weights directly because they are zero.
            We need to at least do an apply_gradients once and then set the optimizer weights.
        """
        # Set the path for the weights and optimizer
        model_weights_path = f"{old_model_dir}/{old_model_file}"
        opt_path   = f"{old_model_dir}/optimizer.pkl"

        # Load the optimizer weights
        with open(opt_path, 'rb') as f:
            opt_weights = pickle.load(f)
        
        # Get the model's trainable weights
        grad_vars = self.model.trainable_weights
        # This need not be model.trainable_weights; it must be a correctly-ordered list of 
        # grad_vars corresponding to how you usually call the optimizer.
        zero_grads = [tf.zeros_like(w) for w in grad_vars]

        # Apply gradients which don't do nothing with Adam
        self.optimizer.apply_gradients(zip(zero_grads, grad_vars))

        # Set the weights of the optimizer
        self.optimizer.set_weights(opt_weights)

        # NOW set the trainable weights of the model
        self.model.load_weights(model_weights_path)


    def _update_summary_logging(self, epoch):
        """
            Tf.summary for epoch level loss
        """
        # Filter out the train and val metrics
        train_metrics = {k.replace('train_',''): v for k, v in self.loss_metrics.items() if k.startswith('train_')}
        val_metrics = {k.replace('val_',''): v for k, v in self.loss_metrics.items() if k.startswith('val_')}
        
        # Summary writer
        with self.train_writer.as_default():
            tf.summary.scalar(f"{self.network_name}/learning_rate", self.optimizer.lr, step=epoch)
            for key in train_metrics.keys():
                tf.summary.scalar(f"{self.network_name}/{key}",  train_metrics[key].result(), step=epoch)         
        
        with self.val_writer.as_default():
            for key in val_metrics.keys():
                tf.summary.scalar(f"{self.network_name}/{key}",  val_metrics[key].result(), step=epoch)
   