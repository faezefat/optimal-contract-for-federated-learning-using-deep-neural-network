import tensorflow as tf
class misreport:
    def __init__(self, config):
        self.theta_min = config.theta_min
        self.theta_max = config.theta_max
        self.num_misreports = config.num_misreports

    # @tf.function  # Compiles this function into a single optimized graph
    def expected_rgt(self, utility_vehicles, utility_vehicles_misrprt):
        # Pre-compute expanded utilities to avoid unnecessary recomputation
        utility_vehicles_expnd = tf.tile(tf.expand_dims(utility_vehicles, 0), [self.num_misreports, 1, 1])
        excess_from_utility = tf.nn.relu(utility_vehicles_misrprt - utility_vehicles_expnd)
        rgt = tf.reduce_mean(tf.reduce_max(excess_from_utility, axis=0), axis=0)
        return rgt

    def assign_var(self, theta_hat_var, theta_hat_var_init):
        # Wrapping assign in tf.function is not beneficial as it runs once per iteration
        return tf.compat.v1.assign(theta_hat_var, theta_hat_var_init)

    # @tf.function  # Graph-compiling the clipping function
    def clip_var(self, theta_hat_var):
        clipped_value = tf.clip_by_value(theta_hat_var, self.theta_min, self.theta_max)
        return tf.compat.v1.assign(theta_hat_var, tf.clip_by_value(theta_hat_var, self.theta_min, self.theta_max))

    def reset_opt_vars(self, vars):
        return tf.compat.v1.variables_initializer(vars)

def update_theta_hat(theta_hat_init, idx, theta_hat_updat):
    theta_hat_init[:, idx, :, :] = theta_hat_updat
    return theta_hat_init
