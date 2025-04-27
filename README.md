# SSBEM-drift-adaptation-model
This is the repository for the paper: **Concept Drift Meets Industrial Data Streams: An Efficient Drift Adaptation Framework with Knowledge Embedding and Transfer**.

**This repository mainly consists of three parts:**

1) the main code for online prediction under incremental drift data;
2) original N-BEATS and proposed SSBEM model architecture;
3) the trained model checkpoints for various comparison models under incremental drift scenarios.

You can refer to the main online prediction code (main_SSBEM_incremental.py) to understand the design of the online data stream, the model prediction logic, and the recursive update strategy for model parameters.

**To evaluate different model checkpoints, please configure the comparison model settings in main_SSBEM_incremental.py:**

#Comparison model setting
#NBEATS: model = 'NBEATS'; model_load_path = 'checkpoints_NBEATS'; recursive_function = False
#SSBEM_B: model = 'SSBEM'; model_load_path = 'checkpoints_SSBEM_B'; recursive_function = False
#SSBEM_BR: model = 'SSBEM'; model_load_path = 'checkpoints_SSBEM_BR'; recursive_function = True
#SSBEM_BRD: model = 'SSBEM'; model_load_path = 'checkpoints_SSBEM_BRD'; recursive_function = True
#SSBEM_BRN: model = 'SSBEM'; model_load_path = 'checkpoints_SSBEM_BRN'; recursive_function = True
#SSBEM_BRS: model = 'SSBEM'; model_load_path = 'checkpoints_SSBEM_BRS'; recursive_function = True
model = 'SSBEM'
model_load_path = 'checkpoints_SSBEM_BRS'
recursive_function = True
