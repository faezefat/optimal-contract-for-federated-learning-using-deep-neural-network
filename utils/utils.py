import os
import numpy as np
import logging
import logging.handlers
import tensorflow as tf
import matplotlib.pyplot as plt
import math
def get_logger(log_dir):
    '''
    Args:
        log_dir (str): Path of the log file.
    '''
    # Create the directory if it does not exist
    if not os.path.exists(os.path.dirname(log_dir)):
        os.makedirs(os.path.dirname(log_dir))

    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid adding multiple handlers
    if not logger.hasHandlers():
        # Create StreamHandler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        # Create FileHandler for saving logs to a file
        file_handler = logging.FileHandler(log_dir, 'a')
        file_handler.setLevel(logging.INFO)

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger

def Optimizer(opt, lr):
    if opt == 'Adam':
        return tf.compat.v1.train.AdamOptimizer(
        learning_rate=lr
        )

    elif opt == 'SGD':
        return tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate = lr
        )
    
def plot_graphs(train_metric, iter_list, expected_value, labels):
    # Adjust the figure size for better spacing and clarity
    num_plots = len(labels)
    
    # Determine grid size: 2 rows for 4 plots, you can adjust this as needed
    num_rows = (num_plots + 1) // 2  # Calculate rows dynamically based on the number of labels
    num_cols = 2 if num_plots > 1 else 1  # If there's only one plot, keep it in one column

    plt.figure(figsize=(15, num_rows * 4))  # Dynamically adjust height based on rows
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Adjust spacing between subplots

    for idx in range(num_plots):
        plt.subplot(num_rows, num_cols, idx + 1)  # Create a subplot in the grid
        plt.plot(iter_list, [item[idx] for item in train_metric], color='g', marker='o', label='Train')

        # Plot the expected value if it exists
        if expected_value[idx] is not None:
            plt.axhline(y=expected_value[idx], color='b', linestyle='--', label='Expected value')

        plt.title(f"{labels[idx]} vs Iteration", fontsize=14)  # Add a title for each plot
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel(labels[idx], fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True)  # Add grid lines for better readability

    plt.show()



def IC_IR_test_graph(type_list, agent_utility_list, actual_type_list):
    plt.subplots_adjust(hspace=0.7)
    num_sample = len(actual_type_list)
    for iter in range(num_sample):
        # the utility og vehicles for different reported type
        plt.plot(type_list[iter], agent_utility_list[iter], label=f"Type data owner is: {actual_type_list[iter]}")
        # type that maximize the utility of vehicle
        indx = type_list[iter][agent_utility_list[iter].index(max(agent_utility_list[iter]))]
        plt.scatter(indx, max(agent_utility_list[iter]), marker="o", color='0', linewidth=3)
        # actual type of vehicle
        indx_actual = type_list[iter].index(actual_type_list[iter])
        plt.scatter(actual_type_list[iter], agent_utility_list[iter][indx_actual], marker='*', linewidth=3)

        # Naming the x-axis, y-axis and the whole graph
        plt.xlabel("Reported Type by Vehicle", fontsize=15)
        plt.ylabel("Utility of Vehicle", fontsize=15)
        plt.legend()
        plt.grid()


    plt.show()

