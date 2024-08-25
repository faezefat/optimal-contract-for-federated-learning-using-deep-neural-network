import tensorflow as tf
class neural_network():
    def __init__(self, config_setting, utilities):
        # Input and output size of neural network
        self.num_input = config_setting.num_pvt_info
        self.num_out = config_setting.num_pvt_info

        # Parameters of alloc and reward networks
        self.num_a_layers = config_setting.num_a_layers
        self.num_a_hidden_units = config_setting.num_a_hidden_units
        self.w_alloc = []
        self.b_alloc = []
        self.num_r_layers = config_setting.num_r_layers
        self.num_r_hidden_units = config_setting.num_r_hidden_units
        self.w_reward = []
        self.b_reward = []

        # Initilizer of weights and biases
        if config_setting.init == "gu" :
            self.w_init = tf.keras.initializers.glorot_uniform()
        self.b_init = tf.keras.initializers.Zeros()

        # Activation function of middle layers
        if config_setting.activation == "tanh" :
            self.activation = tf.tanh
        # Thresholds
        self.f_min =  config_setting.f_min
        self.f_max =  config_setting.f_max

        # use instance of class utilities
        self.utilities = utilities

        # Initialize layers once
        self.net_layers()

    def net_layers(self):
        """
        allocation network parameters:
        """
        with tf.compat.v1.variable_scope("alloc"):
            # Weights of allocation network
            # Input Layer
            self.w_alloc.append(tf.compat.v1.get_variable("w_a_0", shape=[self.num_input, self.num_a_hidden_units], dtype=tf.float32, initializer=self.w_init, trainable=True))

            # Hidden Layers
            for i in range(1, self.num_a_layers - 1):
                wname = "w_a_" + str(i)
                self.w_alloc.append(tf.compat.v1.get_variable(wname, shape=[self.num_a_hidden_units, self.num_a_hidden_units], dtype=tf.float32, initializer=self.w_init, trainable=True))

            # Output Layer
            wname = "w_a_" + str(self.num_a_layers - 1)
            self.w_alloc.append(tf.compat.v1.get_variable(wname, shape=[self.num_a_hidden_units, (self.num_out + 1) * (self.num_out + 1)], dtype=tf.float32, initializer=self.w_init, trainable=True))

            # Biases of allocation network
            for i in range(self.num_a_layers - 1):
                wname = "b_a_" + str(i)
                self.b_alloc.append(tf.compat.v1.get_variable(wname, shape=[self.num_a_hidden_units], dtype=tf.float32, initializer=self.b_init, trainable=True))
            # Last layer
            wname = "b_a_" + str(self.num_a_layers - 1)
            self.b_alloc.append(tf.compat.v1.get_variable(wname, shape=[(self.num_out + 1) * (self.num_out + 1)], dtype=tf.float32, initializer=self.b_init, trainable=True))


        """
        Reward Netwok parameters
        """
        with tf.compat.v1.variable_scope("reward"):
            # Weights of reward network
            # Input Layer
            self.w_reward.append(tf.compat.v1.get_variable("w_r_0", shape=[self.num_input, self.num_r_hidden_units], dtype=tf.float32, initializer=self.w_init, trainable=True))

            # Hidden Layers
            for i in range(1, self.num_r_layers - 1):
                wname = "w_r_" + str(i)
                self.w_reward.append(tf.compat.v1.get_variable(wname, shape=[self.num_r_hidden_units, self.num_r_hidden_units], dtype=tf.float32, initializer=self.w_init, trainable=True))

            # Output Layer
            wname = "w_r_" + str(self.num_r_layers - 1)
            self.w_reward.append(tf.compat.v1.get_variable(wname, shape=[self.num_r_hidden_units, self.num_out], dtype=tf.float32, initializer=self.w_init, trainable=True))

            # Biases of reward network
            for i in range(self.num_r_layers - 1):
                wname = "b_r_" + str(i)
                self.b_reward.append(tf.compat.v1.get_variable(wname, shape=[self.num_r_hidden_units], dtype=tf.float32, initializer=self.b_init, trainable=True))

            wname = "b_r_" + str(self.num_r_layers - 1)
            self.b_reward.append(tf.compat.v1.get_variable(wname, shape=[self.num_out], dtype=tf.float32, initializer=self.b_init, trainable=True))


    def forward(self, theta, theta_org, batch_size):
        """
        Forward section of allocation Network
        """
        # Input
        theta_in = tf.reshape(theta, [-1, self.num_out])
        # Forward part of Allocation
        alloc = tf.matmul(theta_in, self.w_alloc[0]) + self.b_alloc[0]
        alloc = self.activation(alloc)

        for i in range(1, self.num_a_layers - 1):
            alloc = tf.matmul(alloc, self.w_alloc[i]) + self.b_alloc[i]
            alloc = self.activation(alloc)

        alloc = tf.matmul(alloc, self.w_alloc[-1]) + self.b_alloc[-1]
        alloc = tf.reshape(alloc, [-1, self.num_out + 1, self.num_out + 1])
        alloc = tf.sigmoid(alloc, 'alloc_sigmoid')
        alloc = tf.slice(alloc, [0, 0, 0], size=[-1, self.num_out, self.num_out], name='alloc_out')

        # Map the output of allocation network between f_max and f_min
        coef = self.f_max - self.f_min
        const = self.f_min
        alloc_final = const + coef * alloc
        alloc_final = tf.reshape(alloc_final, [-1, batch_size, self.num_out])
        alloc_final = tf.transpose(alloc_final, perm=[1, 0, 2])

        """
        Forward section of Rewrad network
        """
        # Forward part of Reward Network
        reward = tf.matmul(theta_in, self.w_reward[0]) + self.b_reward[0]
        reward = self.activation(reward)

        for i in range(1, self.num_r_layers - 1):
            reward = tf.matmul(reward, self.w_reward[i]) + self.b_reward[i]
            reward = self.activation(reward)

        reward = tf.matmul(reward, self.w_reward[-1]) + self.b_reward[-1]
        reward = tf.sigmoid(reward, 'reward_sigmoid')

        # Map output between 1,2 to satisfy IR constraint within structure
        coef = 2 - 1
        const = 1
        reward = const + coef * reward
        reward = tf.reshape(reward, [-1, batch_size])
        reward = tf.transpose(reward, perm=[1, 0])

        # Obtain necessary tensors for satisfy IR constraint from utilities() class
        # total_cost is the cost of vehicles for consuming their cpu, reward is like multiplier
        # which is greater than one, therefore the rewrd is always greater than total cost, so
        # the utility of vehicle is greater than zero.
        _, total_cost = self.utilities.vehicle_utility(theta_org, alloc_final, reward)
        reward = reward * total_cost

        return alloc_final, reward, theta_in
