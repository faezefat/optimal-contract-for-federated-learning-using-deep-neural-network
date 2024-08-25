class config_setting:

    """
    Federated Learning parameters
    """
    num_vehicles = 20
    num_pvt_info = 1
    distribution_type = "uniform"

    """
    neural network parameters
    """
    # Initialization
    init = "gu"
    # Activation function for middle layers
    activation = "tanh"
    # Hidden layers in alloc and reward networks
    num_a_layers = 2
    num_r_layers = 2
    # Neurons in alloc and reward network
    num_a_hidden_units = 25
    num_r_hidden_units = 7

    """
    Utility Functions parameters
    """
    """ params for RSU utility function """
    # satisfaction degree parameter of RSU
    sat_deg = 800
    # T-max :  RSU’s maximum tolerance time for the FL task.
    T_max = 250
    # T_com : average transmission time to transmit local model updates in a global iteration
    T_com = 10

    """ params for vehicles utility function"""
    # ζi : is the effective capacitance parameter of computing chip set for vehicle i
    cap_param = 1
    # µ :  pre-defined parameter for energy consumption
    energy_cons = 1
    # E_com : energy consumption by vehicle i to transmit local model updates in a global iteration.
    E_com = 20

    """ params for both  utility functions """
    # s_i : is the size of local data sample for vehicle i
    s_i = 23
    # c_i =  number of CPU cycles for a vehicle i
    c_i = 5
    # local model iterations
    psi = 2

    """ Thresholds for Constraints"""
    # Minimum CPU frequency of vehicles
    f_min = 0.5
    # Maximum CPU frequency of vehicles
    f_max = 2
    # Maximun reward from RSU that can allocates to the vehicles
    R_max = 1500


    """
    Augmented lagrangian parameters
    """
    # Initial update rate
    update_rate = 0.0001
    # Initial Lagrange weights for equal constraints (rgt_i = 0)
    w_rgt_init_val = 0.01
    # Initial Lagrange weights for inequal constraints ( sum(R_i) <= total rewad (R_max))
    w_rwd_init_val = 0.001
    # Lagrange update frequency
    update_frequency = 200
    # Value by which update rate is incremented
    up_op_add = 0.1
    # Frequency at which update rate is incremented
    up_op_frequency = 300

    """
    misreport parameters (Calculate best misreport that maximize utility of vehicles)
    """
    # Cache-misreports after misreport optimization
    adv_reuse = True
    # Number of misreport initialization for training
    num_misreports = 1
    # Number of steps for misreport computation
    gd_iter = 20
    # Learning rate of misreport computation
    gd_lr = 0.1

    """ train dataset parameters """
    # min value of data
    theta_min = 3.92
    # maximum value of data
    theta_max = 48.78
    # Number of batches
    num_batches = 500
    # Train batch size
    batch_size = 128
    # total number of train dataset
    num_instances = num_batches * batch_size

    """ train parameters """
    # Random seed
    seed = 42
    # Iter from which training begins. If restore_iter = 0 for default. restore_iter > 0 for starting
    # training form restore_iter [needs saved model]
    restore_iter = 0
    # max iters to train
    max_iter = 15000
    # Learning rate of network param updates
    learning_rate = 2e-3

    """ validation dataset parameters """
    # Number of validation batches
    val_num_batches = 10
    # total number of validation dataset
    val_num_instances = val_num_batches * batch_size
    val_num_misreports = 1


    """ Test dataset parameters """
    # Restore the iter_model for evaluation
    test_restore_iter = 15000
    # Number of test batches
    test_num_batches = 1
    # Test batch size
    test_batch_size = 1
    # Misreport numbers for test data
    test_num_misreports = 1
    # Total instances for test
    test_num_instances = test_num_batches * test_batch_size

    """ saving models """
    # Direction for saving the results of training
    dir_name = 'DNN_Contract_FL'
    # Set the training will start from iteration 0 or from other model stored at restore_iter_train
    restore_iter_train = 0

    """ Resume model """
    resume_model = None

    """ Parameters for saving model """
    # Number of models to store on disk
    max_to_keep = 25
    # Frequency at which models are saved-
    save_iter = 20
    # Train stats print frequency
    print_iter = 10

    """ Seeds """
    # Test Seed
    test_seed = 100

