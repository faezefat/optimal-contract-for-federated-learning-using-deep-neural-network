# Optimal Contract for Federated Learning Using Deep Neural Networks

## Description

This project focuses on designing an optimal contract within a federated learning framework for vehicular networks. In this scenario:

- The RSU (RoadSide Unit) serves as the task publisher.
- Vehicles act as data owners, training local models based on the given task and sending their trained models back to the RSU for aggregation into a global model.

### Problem Background

In the federated learning setup within vehicular networks:

- Vehicles use their local computational resources for model training. However, since they cannot utilize these resources for other purposes during this process, some vehicles may choose not to participate.
- To encourage participation, the RSU offers rewards based on the data quality used by each vehicle for training. Since vehicles report their own data quality, which is private information, they might misreport to gain higher rewards.

### Objective

The goal is to design a contract that specifies:

- **\( X_i \)**: The computational resource the vehicle must dedicate to local model training.
- **\( R_i \)**: The reward allocated for this training and resource usage.

The contract must be designed to:

1. Maximize the utility of the RSU.
2. Ensure vehicles receive non-negative utility (Individual Rationality).
3. Encourage truthful reporting of private information (Incentive Compatibility).
4. Ensure that the total reward distributed does not exceed a maximum allowable reward.
5. Ensure that the computational resource \( X_i \) is within specified frequency bounds.

### Problem Formulation

The problem of finding the optimal contract can be summarized as follows:

<p align="center">
  <img src="https://quicklatex.com/cache3/85/ql_00035d45896495eca030c03f58e75e85_l3.png" alt="Formula">
</p>

To solve this optimization problem, we will use deep neural networks. Specifically, we will utilize two neural networks to determine \( X_i \) and \( R_i \). We will adjust the network architecture to incorporate the constraints of (1b) and (1c) within the neural network structure and reformulate the other constraints to enable training. Finally, we will define a loss function using the augmented Lagrangian method, with the goal of minimizing this loss through the neural network.

### Implementation

This project is implemented using the following packages:

- TensorFlow
- Python
- NumPy
- Matplotlib

### How to Run the Code

1. Clone the repository or download the code files.
2. Open the code in Python.
3. Run the following commands in your terminal:

```bash
!python train.py
!python test.py

### Evaluation Results

The results for evaluating our proposed model are shown below:

<p align="center">
  <img src="path/to/your/image.png" alt="Evaluation Results">
</p>
