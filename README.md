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

**
\begin{subequations} 
\label{eq:contract_opt} 
\begin{align} 
& \max_{(R(\theta_i), x(\theta_i))} U_{TP}\big(x(\theta_i), R(\theta_i)\big) \tag{\ref{eq:contract_opt}},\\
& \text{s.t.} \quad \sum_{i \in \mathcal{N}}\mathbb{E}_{{\theta_i}}[R({\theta}_i)] \leq R_{max},\\
& \quad f^{min}_i \leq x(\theta_i) \leq f^{max}_i \label{eq:resorce_limit},\\
& \quad U_i^{D}\big(\theta_i,R(\theta_i), x(\theta_i)\big) \geq 0, \quad \forall \theta \in [\underline{\theta}, \overline{\theta}], i \in \mathcal{N} \label{eq:IR-con},\\  
& \quad U_i^{D}\big(\theta_i, R(\theta_i), x(\theta_i)\big) \geq U_i^{D}\big(\theta_i, R(\hat{\theta}_i), x(\hat{\theta}_i)\big) \label{eq:IC-con},\\
& \quad \forall \theta_i, \hat{\theta}_i \in [\underline{\theta}, \overline{\theta}], i \in \mathcal{N}.
\end{align}
\end{subequations}
**

