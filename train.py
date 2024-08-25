import os
import sys
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import time
from tqdm import tqdm
import random

from utils import utils
from utils.utils import Optimizer
from data_gent.data_generator import data_generator
from utility.utility import utilities
from models.NN_model import neural_network
from models.regret import misreport
from models import regret
from loss.loss import Loss
from config.config_setting import config_setting
# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.compat.v1.set_random_seed(42)

tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

def main(config):
    print('#----------Creating logger----------#')
    # 1-Create output-dir
    if not os.path.exists(config.dir_name): os.mkdir(config.dir_name)

    # Determine the model for start of training
    log_suffix = '_' + str(config.restore_iter_train) if config.restore_iter_train > 0 else ''
    log_fname = os.path.join(config.dir_name, 'train' + log_suffix + '.txt')

    global logger
    logger = utils.get_logger(log_fname)
    logger.info("Logger is set up and ready!")

    print('#----------Preparing dataset----------#')
    train_dg = data_generator(config, config.num_instances)
    val_dg = data_generator(config, config.val_num_instances)
    theta_t_train, theta_hat_t_init_train = train_dg.generate_data()
    theta_t_val, theta_hat_t_init_val = val_dg.generate_data()

    print('#----------Preparing variables, placeholders----------#')
    # Shape of variables and placeholders
    theta_shape = [config.batch_size, config.num_vehicles, config.num_pvt_info]
    theta_hat_shape = [config.num_vehicles, config.num_misreports, config.batch_size, config.num_vehicles, config.num_pvt_info]
    theta_hat_var_shape = [config.num_misreports, config.batch_size, config.num_vehicles, config.num_pvt_info]

    # placeholders for theta as input of neural network and theta_hat as misreport
    theta_var = tf.compat.v1.placeholder (tf.float32, shape=theta_shape, name='theta')
    theta_hat_init_var = tf.compat.v1.placeholder (tf.float32, shape=theta_hat_var_shape, name='theta_hat_init')

    # Create variables which will be optimized during training
    # variable for finding best misreport
    with tf.compat.v1.variable_scope('theta_hat_var', reuse=tf.compat.v1.AUTO_REUSE):
        theta_hat_var = tf.compat.v1.get_variable('theta_hat_var', shape = theta_hat_var_shape, dtype = tf.float32)

    # Augmented Lagrangian variables
    with tf.compat.v1.variable_scope('w_rwd', reuse=tf.compat.v1.AUTO_REUSE):
        w_rwd = tf.Variable(config.w_rwd_init_val, 'w_rwd')
    with tf.compat.v1.variable_scope('w_rgt'):
        w_rgt = tf.Variable(np.ones(config.num_vehicles).astype(np.float32)*config.w_rgt_init_val, 'w_rgt')
    update_rate = tf.Variable(config.update_rate, trainable = False)




    print('#----------Prepareing Models----------#')
    # Initialize the models
    utility_funcs = utilities(config)
    dnn_model = neural_network(config, utility_funcs)
    misrprt_cal = misreport(config)
    alloc, reward, theta_in = dnn_model.forward(tf.transpose(theta_var, perm=[1, 0, 2]), theta_var, config.batch_size)
    alloc_misrprt, reward_misrprt, theta_in_misrprt = dnn_model.forward(tf.transpose(theta_hat_var, perm=[3, 2, 1, 0]), theta_hat_var, config.batch_size)


    # Neural Network Variable Lists
    alloc_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='alloc')
    reward_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='reward')
    var_list = alloc_vars + reward_vars

    # Calculate utilities for truthful report and misreport
    utility_vehicles, _ = utility_funcs.vehicle_utility(theta_var, alloc, reward)
    utility_RSU = utility_funcs.rsu_utility(theta_var, alloc, reward)
    utility_vehicles_misrprt, _ = utility_funcs.vehicle_utility(theta_var, alloc_misrprt, reward_misrprt)

    # Calculate expected ex-post regret
    rgt = misrprt_cal.expected_rgt(utility_vehicles, utility_vehicles_misrprt)

    print('#----------Prepareing loss, Optimizer----------#')
    # Losses
    losses = Loss(config, utility_funcs, dnn_model)
    rwd_loss = losses.reward_loss(reward, w_rwd, update_rate, config.batch_size)
    lag_loss, rgt_penalty = losses.lag_loss(rgt, w_rgt, update_rate)
    net_loss = utility_RSU - lag_loss - rgt_penalty - rwd_loss
    misrprt_loss = tf.reduce_sum(utility_vehicles_misrprt)
    penalty = lag_loss + rgt_penalty + rwd_loss

    # Optimizers
    train_opt = utils.Optimizer("Adam", config.learning_rate).minimize(-net_loss, var_list=var_list)
    opt_misrprt = utils.Optimizer("Adam", config.gd_lr)
    train_misrprt_opt = opt_misrprt.minimize(-misrprt_loss, var_list=[theta_hat_var])
    lagrange_opt = utils.Optimizer("SGD", update_rate).minimize(-lag_loss, var_list = [w_rgt])
    rwd_opt = utils.Optimizer("SGD", update_rate).minimize(-rwd_loss, var_list = [w_rwd])
    increment_update_rate = update_rate.assign(update_rate + config.up_op_add)

    print('#----------Set other params----------#')
    rgt_mean = tf.reduce_mean(rgt)
    reward_mean = tf.reduce_sum(tf.reduce_mean(reward, axis=0))
    irp_mean = tf.reduce_mean(tf.nn.relu(-utility_vehicles))

    # Metrics
    reward_mean = tf.reduce_sum(tf.reduce_mean(reward, axis=0))
    metrics = [-net_loss, utility_RSU, rgt_mean, reward_mean, penalty, irp_mean, update_rate]
    metric_names = ["net_loss", "RSU utility", "regret", "reward_mean", "penalty", "irp_mean"]

    print('#----------Training----------#')
    train_metric = []
    iter_list = []
    saver = tf.compat.v1.train.Saver(max_to_keep = config.max_to_keep)
    iter = config.restore_iter

    # utils for calculating regret
    clip_var = misrprt_cal.clip_var(theta_hat_var)
    reset_opt = misrprt_cal.reset_opt_vars(opt_misrprt.variables())
    assign_var = misrprt_cal.assign_var(theta_hat_var, theta_hat_init_var)


    # Check if all variables are initialized
    sess = tf.compat.v1.InteractiveSession()
    tf.compat.v1.global_variables_initializer().run()
    train_writer = tf.compat.v1.summary.FileWriter(config.dir_name, sess.graph)

    # all variables
    variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    time_elapsed = 0.0
    with tqdm(total=config.max_iter) as pbar:
        while iter < (config.max_iter):
            tic = time.time()
            # Get a mini-batch
            theta_batch, theta_hat_batch, perm = next(train_dg.data_btach(theta_t_train, theta_hat_t_init_train))

            # update augmented lagrangian param
            if iter == 0:
                sess.run(lagrange_opt, feed_dict = {theta_var:theta_batch})
                sess.run(rwd_opt, feed_dict = {theta_var:theta_batch})
                new_w_rwd = sess.run(tf.nn.relu(variables[1]))
                sess.run(variables[1].assign(new_w_rwd))

            # Get Best Mis-report
            sess.run(assign_var, feed_dict = {theta_hat_init_var:theta_hat_batch})
            for _ in range(config.gd_iter):
                sess.run(train_misrprt_opt, feed_dict = {theta_var:theta_batch})
                sess.run(clip_var)
            sess.run(reset_opt)
            if config.adv_reuse:
                regret.update_theta_hat(theta_hat_t_init_train, perm, sess.run(theta_hat_var))

            # Update network params
            sess.run(train_opt, feed_dict = {theta_var:theta_batch})

            iter += 1
            # Update Augmented Lagrangian parameters
            if iter % config.update_frequency == 0:
                sess.run(lagrange_opt, feed_dict = {theta_var:theta_batch})

            if iter % config.up_op_frequency == 0:
                sess.run(increment_update_rate)

            # Calculate the running time
            toc = time.time()
            time_elapsed += (toc - tic)

            # Save model
            if ((iter % config.save_iter) == 0) or (iter == config.max_iter):
                saver.save(sess, os.path.join(config.dir_name,'model'), global_step = iter)

            # Print the results
            if (iter % config.print_iter) == 0:
                # Train Set Stats
                metric_vals = sess.run(metrics, feed_dict = {theta_var:theta_batch})
                train_metric.extend([metric_vals])
                iter_list.extend([iter])
                fmt_vals = tuple([ item for tup in zip(metric_names, metric_vals) for item in tup ])
                log_str = "TRAIN-BATCH Iter: %d, t = %.4f"%(iter, time_elapsed) + ", %s: %.6f"*len(metric_names)%fmt_vals
                logger.info(log_str)

    return train_metric, iter_list


if __name__ == '__main__':
    config = config_setting
    train_metric, iter_list = main(config)
    ["net_loss", "RSU utility", "regret", "reward_mean", "penalty", "irp_mean"]
    labels = ["Augmenred Lagrangian Loss", "Utility of RSU", "Mean Regret of All Vehicles", "Total Reward of All Vehicles"]
    expected_value = [None, None, 0, config.R_max]
    utils.plot_graphs(train_metric, iter_list, expected_value, labels)





