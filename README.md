# optimal-contract-for-federated-learning-using-deep-neural-network
# Descreption:
The goal of this  project to find the optimal contract within federated learning framework in vehicular network. In Federated learning within vehicular network, RSU (RoadSideUnit) acts as task publisher, and vehicles acts as dataowners which they will train their local model based on the task and send their trained local model to RSU to aggragate them and create global model, and this loop will continue. As vehilces use their local sources for training and they can not use their sources for other things, they might not participate in training of model. The RSU will allocate reward to them to persuade them to participate but this reward is based on the data quality of each vehicles use to train their local model. The vehicles report their data quality which is their private information and recieve the reward but the vehicles may report their to recieve more reawrds. The RSU needs to design the contract which this contract contains \[(X_i, R_i)\], $X_i$ is the computational resource that veicle must train the local model with this frequency and $R_i$ is allocated reward for training model and using local resources,  which in this contract maximize its utility while ensure the vehicles gtes non negative utility (Individual Rationality) and report their type truthfully (Incentive Compatibility). The problem of finding otimal contract is summerized as follows:
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
