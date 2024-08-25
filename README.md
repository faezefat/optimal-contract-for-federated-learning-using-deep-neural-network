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
<p align="center">
  <img src="https://quicklatex.com/cache3/85/ql_00035d45896495eca030c03f58e75e85_l3.png" alt="Formula">
</p>


