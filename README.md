# Optimal Contract for Federated Learning Using Deep Neural Networks

## Description

This project focuses on designing an optimal contract within a federated learning framework for vehicular networks. In this scenario:

- The RSU (RoadSide Unit) serves as the task publisher.
- Vehicles act as data owners, training local models based on the given task and sending their trained models back to the RSU for aggregation into a global model.

### Problem Background

In the federated learning setup within vehicular networks:

- Vehicles use their local computational resources for model training. However, since they can't utilize these resources for other purposes during this process, some vehicles may choose not to participate.
- To encourage participation, the RSU offers rewards based on the data quality used by each vehicle for training. Since vehicles report their own data quality, which is private information, they might misreport to gain higher rewards.

### Objective

The goal is to design a contract that specifies:

- **\( X_i \)**: The computational resource the vehicle must dedicate to local model training.
- **\( R_i \)**: The reward allocated for this training and resource usage.

The contract must be designed to:

1. Maximize the utility of the RSU.
2. Ensure vehicles receive non-negative utility (Individual Rationality).
3. Encourage truthful reporting of private information (Incentive Compatibility).

### Problem Formulation
The problem of finding the optimal contract can be summarized as follows:
![Formula](https://latex.codecogs.com/png.latex?\begin%7Bsubequations%7D%20%5Clabel%7Beq%3Acontract_opt%7D%20%5Cbegin%7Balign%7D%20%26%5Cmax_%7B(R(%5Ctheta_i)%2C%20x(%5Ctheta_i))%7D%20U_%7BTP%7D%5Cbig(x(%5Ctheta_i)%2C%20R(%5Ctheta_i)%5Cbig)%5C%5C%20%26%20%5Ctext%7Bs.t.%7D%5C%2C%20%5Csum_%7Bi%5Cin%5Cmathcal%7BN%7D%7D%5Cmathbb%7BE%7D_%7B%7B%5Ctheta_i%7D%7D%5BR(%7B%5Ctheta%7D_i)%5D%20%5Cleq%20R_%7Bmax%7D%2C%5C%5C%20%26%20%5C%2C%20f%5E%7Bmin%7D_i%20%5Cleq%20x(%5Ctheta_i)%20%5Cleq%20f%5E%7Bmax%7D_i%2C%5C%5C%20%26%20%5C%2C%20U_i%5E%7BD%7D%5Cbig(%5Ctheta_i%2CR(%5Ctheta_i)%2C%20x(%5Ctheta_i)%5Cbig)%20%5Cgeq%200%2C%20%5Cforall%20%5Ctheta%20%5Cin%20%5B%5Cunderline%7B%5Ctheta%7D%2C%20%5Coverline%7B%5Ctheta%7D%5D%20%2Ci%20%5Cin%20%5Cmathcal%7BN%7D%2C%5C%5C%20%26%20%5C%2C%20U_i%5E%7BD%7D%5Cbig(%5Ctheta_i%2C%20R(%5Ctheta_i)%2C%20x(%5Ctheta_i)%5Cbig)%20%5Cgeq%20U_i%5E%7BD%7D%5Cbig(%5Ctheta_i%2C%20R(%5Chat%5Ctheta_i)%2C%20x(%5Chat%5Ctheta_i)%5Cbig)%2C%20%5Cforall%20%5Ctheta_i%2C%20%5Chat%5Ctheta_i%20%5Cin%20%5B%5Cunderline%7B%5Ctheta%7D%2C%20%5Coverline%7B%5Ctheta%7D%5D%20%2Ci%20%5Cin%20%5Cmathcal%7BN%7D.%5Cend%7Balign%7D%20%5Cend%7Bsubequations%7D)

