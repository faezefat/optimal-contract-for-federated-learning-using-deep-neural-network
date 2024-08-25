import tensorflow as tf
class Loss():
    def __init__(self, config_setting, utilities, neural_network):
        self.R_max = config_setting.R_max
        self.utilities = utilities
        self.neural_network = neural_network

    def misreport_loss(self, theta_hat_var, batch_size):
        # theta_hat_var is a variable for calculating regret
        # here we calculate allocation and reward when the vehicles report any possible type
        alloc_misrprt, reward_misrprt, _ = \
        self.neural_network.forward(tf.transpose(theta_hat_var, perm=[3, 2, 1, 0]), theta_hat_var, batch_size)
        # we want to minimize the utility that the vehicles can get by misreporting the type
        utility_misrprt = self.utilities.vehicle_utility(theta_hat_var, alloc_misrprt, reward_misrprt)
        loss_misrprt = tf.reduce_sum(utility_misrprt)
        return loss_misrprt


    def reward_loss(self, reward, w_rwd, update_rate, batch_size):
        R_max_vect = tf.constant(self.R_max, dtype=tf.float32, shape=(batch_size,), name='R_max_vect')
        rwd_loss = (tf.math.square(tf.nn.relu(update_rate * tf.reduce_mean(tf.reduce_sum(reward, axis=1) - R_max_vect, axis=0) + w_rwd))\
                    - tf.math.square(w_rwd))  / (2.0 * update_rate)
        # Acording to augmented lagrangian method while the total_loss must minimized the loss for
        # other parties must maximized to get optimal values for augmented parameters
        return rwd_loss

    def lag_loss(self, rgt, w_rgt, update_rate):
        lag_loss = tf.reduce_sum(w_rgt * rgt)
        rgt_penalty = update_rate * tf.reduce_sum(tf.square(rgt)) / 2.0
        return lag_loss, rgt_penalty


