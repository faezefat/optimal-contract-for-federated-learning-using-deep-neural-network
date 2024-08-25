import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")

from utils import utils
from utils.utils import IC_IR_test_graph
from utility.utility import utilities
from models.NN_model import neural_network
from config.config_setting import config_setting

tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

def test(config):
    print('#----------Preparing dataset----------#')
    # Test_data
    X_test = np.array([[4, 5, 8, 10, 12, 15, 20, 25, 30, 48, 7, 11, 17, 19, 23, 29, 31, 37, 41, 43]], dtype=np.float32)
    X_test = np.reshape(X_test, [config.test_batch_size, config.num_vehicles, config.num_pvt_info])
    print('#----------Preparing variables, placeholders----------#')
    theta_test_shape = [config.test_batch_size, config.num_vehicles, config.num_pvt_info]
    # Placeholders
    theta_rprt = tf.compat.v1.placeholder (tf.float32, shape=theta_test_shape, name='theta_rprt')
    theta_true = tf.compat.v1.placeholder (tf.float32, shape=theta_test_shape, name='theta_true')

    print('#----------Prepareing Models----------#')
    utility_funcs = utilities(config)
    dnn_model = neural_network(config, utility_funcs)
    # Allocation and Reward or given theta_test(might not be equal to true theta)
    alloc_t, reward_t, theta_in_t = dnn_model.forward(tf.transpose(theta_rprt, perm=[1, 0, 2]), theta_rprt, config.test_batch_size)
    # Calculate vehicles utility(here the vehicles use their true type tu calculate utility)
    vehicles_utility_test,_ = utility_funcs.vehicle_utility(theta_true, alloc_t, reward_t)

    print('#----------Set other params----------#')
    vehicle_idx_list = [0,3,5,7,9]
    num_sample = 5
    actual_type_list = []
    vehicle_utility_list = [[] for _ in range(num_sample)]
    type_list = [[] for _ in range(num_sample)]


    print('#----------Test IC, IR----------#')
    # Initialize variables
    iter = config.test_restore_iter
    dir_name = config.dir_name
    sess = tf.compat.v1.InteractiveSession()
    #reloading model in session
    saver = tf.compat.v1.train.Saver()
    model_path = os.path.join(dir_name,'model-' + str(iter))
    saver.restore(sess, model_path)
    X_org = np.copy(X_test)  # Use numpy's copy
    # Test IC, IR based on original definition

    for iter in tqdm(range(num_sample)):
        vehicle_idx = vehicle_idx_list[iter]
        true_type = X_test[0][vehicle_idx][0]  # Extract the true type using numpy
        actual_type_list.extend([true_type])
        type_list[iter].extend([true_type])
        type_list[iter].extend(np.arange(4, 48, 0.3))
        type_list[iter].sort()

        for pvt_type in type_list[iter]:
            X_test[0][vehicle_idx][0] = pvt_type  # Modify the numpy array directly

            # Convert the numpy array to TensorFlow tensor only when needed
            X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
            X_org_tensor = tf.convert_to_tensor(X_org, dtype=tf.float32)

            vehicle_utility_list[iter].extend([sess.run(vehicles_utility_test[0][vehicle_idx],\
                                                        feed_dict = {theta_rprt:X_test, theta_true:X_org})])
    return type_list, vehicle_utility_list, actual_type_list


config = config_setting
type_list, vehicle_utility_list, actual_type_list = test(config)
IC_IR_test_graph(type_list, vehicle_utility_list, actual_type_list)



