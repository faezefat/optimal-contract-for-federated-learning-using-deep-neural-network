import tensorflow as tf
class utilities():
      def __init__(self, config_setting):
            # Config settings (FL)
            self.sat_deg = config_setting.sat_deg
            self.T_max = config_setting.T_max
            self.T_com = config_setting.T_com
            self.cap_param = config_setting.cap_param
            self.energy_cons = config_setting.energy_cons
            self.E_com = config_setting.E_com
            self.s_i = config_setting.s_i
            self.c_i = config_setting.c_i
            self.psi = config_setting.psi

      # TensorFlow constants
      def init_tensors(self, theta, alloc):
            self.max_time = tf.constant(self.T_max, dtype=tf.float32, shape=theta.shape, name='T_max')
            self.size_local_data = tf.constant(self.s_i, dtype=tf.float32, shape=theta.shape, name='local_sample_size')
            self.cpu_cycle = tf.constant(self.c_i, dtype=tf.float32, shape=theta.shape, name='num_cpu_cycles')
            self.transmission_time = tf.constant(self.T_com, dtype=tf.float32, shape=theta.shape, name='avg_transmission_time')
            self.transmission_energy = tf.constant(self.E_com, dtype=tf.float32, shape=alloc.shape, name='avg_transmission_energy')

      # Compute Satisfaction function
      def sat_comput(self, theta, alloc):
            """ Given private information (theta) and allocation (alloc), computes satisfaction
            Input params:
                  alloc(allocation): [batch_size, num_agents, 1]
                  theta(private_information): [batch_size, num_agents, num_pvt_info]
            Output params:
                  satisfaction: [batch_size, num_agents]
            """
            self.init_tensors(theta, alloc)
            cmp_time = self.size_local_data * self.cpu_cycle / alloc
            total_time = self.psi * cmp_time / theta + self.transmission_time
            return self.sat_deg * tf.math.log(self.max_time - total_time)

      # Compute Utility of RSU (Task Publisher)
      def rsu_utility(self, theta, alloc, reward):
            """ Given private information (theta), reward (reward) and allocation (alloc), computes RSU utility
            Input params:
                  theta: [batch_size, num_agents, num_pvt_info]
                  allocation: [batch_size, num_agents, 1]
                  reward: [batch_size, num_agents]
            Output params:
                  RSU utility: [scaler]
            """
            sat = self.sat_comput(theta, alloc)
            return tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(sat, axis=-1) - reward, axis = 0))

      # Compute Utility of Vehicles (Data Owners)
      def vehicle_utility(self, theta, alloc, reward):
            """ Given input private information (theta), reward (reward) and allocation (alloc), computes agents utility
                  Input params:
                  theta: [batch_size, num_agents, num_pvt_info]
                  alloc: [batch_size, num_agents, 1]
                  reward: [batch_size, num_agents]
                  Output params:
                  utility: [batch_size, num_agents]
            """
            self.init_tensors(theta, alloc)
            cmp_energy = self.cap_param * self.cpu_cycle * self.size_local_data * tf.math.square(alloc)
            total_energy = self.psi * cmp_energy / theta + self.transmission_energy
            total_energy  = tf.reduce_sum(total_energy, axis=-1)
            return reward - self.energy_cons * total_energy, self.energy_cons * total_energy