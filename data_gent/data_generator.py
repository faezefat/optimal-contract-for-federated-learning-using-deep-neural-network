import numpy as np
class data_generator():
    def __init__(self, config_setting, num_instances):
        self.num_vehicles = config_setting.num_vehicles
        self.batch_size = config_setting.batch_size
        self.num_pvt_info = config_setting.num_pvt_info
        self.num_misreports = config_setting.num_misreports
        self.theta_max = config_setting.theta_max
        self.theta_min = config_setting.theta_min
        self.num_instances = num_instances
    def generate_data(self):
        # Here we generate all samples for training and validation
        theta_t_shape = [self.num_instances, self.num_vehicles, self.num_pvt_info]
        theta_hat_t_shape = [self.num_misreports, self.num_instances, self.num_vehicles, self.num_pvt_info]
        theta_t = np.random.uniform(self.theta_min, self.theta_max , size = theta_t_shape)
        # initialization of theta_hat for finding misreport that maximize utility of vehicles
        theta_hat_t_init =  np.random.uniform(self.theta_min, self.theta_max , size = theta_hat_t_shape)      
        return theta_t, theta_hat_t_init

    def data_btach(self, theta_t, theta_hat_t_init):
        # Here we make train data set into batches 
        i = 0
        perm = np.random.permutation(self.num_instances)
        while True:
            idx = perm[i * self.batch_size: (i + 1) * self.batch_size]
            yield theta_t[idx], theta_hat_t_init[:, idx, :, :], idx
            i += 1
            if(i * self.batch_size == self.num_instances):
                i = 0
                perm = np.random.permutation(self.num_instances)
            else: 
                perm = np.arange(self.num_instances)
            

