#!/usr/bin/env python
# coding: utf-8

# ### testing the XGBoost Diabetes example to transpile
# ##### used conda env giza from ll laptop

# ## Create and Train an LR Model
# ### We'll start by creating a simple XGBoost model using Scikit-Learn and train it on ship ETA dataset

# In[2]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from giza.datasets import DatasetsLoader
from giza.agents import GizaAgent
from giza.zkcook import serialize_model

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns



# In[3]:


data  = pd.read_csv('BETF-Final.csv')
data


# In[4]:


data.shape


# In[5]:


#X, y = data.data, data.target

# Drop rows with missing values
df_cleaned = data.dropna(subset=['TotalETF'])

# Prepare your data
X = df_cleaned[['TotalETF']]
y = df_cleaned['ClosingPrice']


# In[6]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


# In[11]:


X_test


# In[7]:


predictions = model.predict(X_test)


# In[8]:


predictions


# In[10]:


value = 453.1

# Reshape into a 2D array with a single feature
reshaped_value = np.array(value).reshape(-1, 1)

# Make the prediction
predictions2 = model.predict(reshaped_value)
print(predictions2)


# In[12]:


reshaped_value


# In[14]:


inputTotalETF = 453.1
input0 = np.array([[inputTotalETF]]).astype(np.float32)
input0


# In[16]:


input0.flatten()[0]


# In[6]:


get_ipython().system('pip install skl2onnx')


# In[7]:


from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Define the initial types for the ONNX model
#initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]  ]))]

initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]  ]))]

# Convert the scikit-learn model to ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save the ONNX model to a file
with open("linear_regression-betf6.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())


# ## Save the model
# ### Save the model in Json format

# ## Transpile your model to Orion Cairo
# ### We will use Giza-CLI to transpile our saved model to Orion Cairo.

# $ giza transpile linear_regression-betf4.onnx --output-path verifiable_betf_lr4
# [giza][2024-06-05 17:29:54.869] No model id provided, checking if model exists ✅
# [giza][2024-06-05 17:29:54.880] Model name is: linear_regression-betf
# [giza][2024-06-05 17:29:55.610] Model Created with id -> 724! ✅
# [giza][2024-06-05 17:29:56.365] Version Created with id -> 1! ✅
# [giza][2024-06-05 17:29:56.375] Sending model for transpilation ✅
# [giza][2024-06-05 17:30:29.406] Transpilation is fully compatible. Version compiled and Sierra is saved at Giza ✅
# [giza][2024-06-05 17:30:31.389] Downloading model ✅
# [giza][2024-06-05 17:30:31.430] model saved at: verifiable_betf_lr

# In[ ]:


$ giza transpile linear_regression-betf4.onnx --output-path verifiable_betf_lr4
[giza][2024-06-06 18:04:42.051] No model id provided, checking if model exists ✅
[giza][2024-06-06 18:04:42.058] Model name is: linear_regression-betf4
[giza][2024-06-06 18:04:42.599] Model Created with id -> 732! ✅
[giza][2024-06-06 18:04:43.332] Version Created with id -> 1! ✅
[giza][2024-06-06 18:04:43.337] Sending model for transpilation ✅
[giza][2024-06-06 18:05:27.067] Transpilation is fully compatible. Version compiled and Sierra is saved at Giza ✅
[giza][2024-06-06 18:05:28.039] Downloading model ✅
[giza][2024-06-06 18:05:28.064] model saved at: verifiable_betf_lr4


# $ giza transpile linear_regression-betf6.onnx --output-path verifiable_betf_lr6
# [giza][2024-06-07 14:06:03.167] No model id provided, checking if model exists ✅
# [giza][2024-06-07 14:06:03.177] Model name is: linear_regression-betf6
# [giza][2024-06-07 14:06:03.910] Model Created with id -> 739! ✅
# [giza][2024-06-07 14:06:04.731] Version Created with id -> 1! ✅
# [giza][2024-06-07 14:06:04.738] Sending model for transpilation ✅
# [giza][2024-06-07 14:06:48.289] Transpilation is fully compatible. Version compiled and Sierra is saved at Giza ✅
# [giza][2024-06-07 14:06:49.131] Downloading model ✅
# [giza][2024-06-07 14:06:49.163] model saved at: verifiable_betf_lr6

# ## Deploy an inference endpoint # 1
# ### Now that our model is transpiled to Cairo we can deploy an endpoint to run verifiable inferences. We will use Giza CLI again to run and deploy an endpoint. Ensure to replace model-id and version-id with your ids provided during transpilation.
# 

# $ giza endpoints deploy --model-id 724 --version-id 1
# ▰▰▰▰▰▰▰ Creating endpoint!
# [giza][2024-06-05 17:33:42.606] Endpoint is successful ✅
# [giza][2024-06-05 17:33:42.618] Endpoint created with id -> 285 ✅
# [giza][2024-06-05 17:33:42.623] Endpoint created with endpoint URL: https://endpoint-giza1-724-1-79763323-7i3yxzspbq-ew.a.run.app 🎉
# 

# $ giza endpoints deploy --model-id 732 --version-id 1
# [giza][2024-06-07 13:52:42.178] Endpoint for model id 732 and version id 1 already exists! ✅
# [giza][2024-06-07 13:52:42.188] Endpoint id -> 312 ✅
# [giza][2024-06-07 13:52:42.190] You can start doing inferences at: None 🚀

# $ giza endpoints deploy --model-id 739 --version-id 1
# ▰▱▱▱▱▱▱ Creating endpoint!
# [giza][2024-06-07 14:09:34.112] Endpoint is successful ✅
# [giza][2024-06-07 14:09:34.118] Endpoint created with id -> 314 ✅
# [giza][2024-06-07 14:09:34.120] Endpoint created with endpoint URL: https://endpoint-giza1-739-1-a8a598d9-7i3yxzspbq-ew.a.run.app 🎉

# ## Create Agent
# ### (you can create this after running verifiable inference, get and download proof and verify proof - need before agent run only)

# giza agents create --model-id 724 --version-id 1 --endpoint-id 285 --name BETF1 --description BETF1
# $ giza agents create --model-id 724 --version-id 1 --endpoint-id 285 --name BETF1 --description BETF1
# [giza][2024-06-06 17:45:28.697] Creating agent ✅
# [giza][2024-06-06 17:45:28.705] Using endpoint id to create agent, retrieving model id and version id
# [giza][2024-06-06 17:45:29.315] Select an existing account to create the agent.
# [giza][2024-06-06 17:45:29.318] Available accounts are:
# ┌──────────┐
# │ Accounts │
# ├──────────┤
# │  giza1   │
# └──────────┘
# Enter the account name: giza1
# {
#   "id": 71,
#   "name": "BETF1",
#   "description": "BETF1",
#   "parameters": {
#     "model_id": 724,
#     "version_id": 1,
#     "endpoint_id": 285,
#     "account": "giza1"
#   },
#   "created_date": "2024-06-07T00:45:34.144701",
#   "last_update": "2024-06-07T00:45:34.144701"
# }

# $ giza agents create --model-id 732 --version-id 1 --endpoint-id 312 --name BETF4 --description BETF4
# [giza][2024-06-07 13:55:43.554] Creating agent ✅
# [giza][2024-06-07 13:55:43.564] Using endpoint id to create agent, retrieving model id and version id
# [giza][2024-06-07 13:55:44.077] Select an existing account to create the agent.
# [giza][2024-06-07 13:55:44.082] Available accounts are:
# ┌──────────┐
# │ Accounts │
# ├──────────┤
# │  giza1   │
# └──────────┘
# Enter the account name: giza1
# {
#   "id": 72,
#   "name": "BETF4",
#   "description": "BETF4",
#   "parameters": {
#     "model_id": 732,
#     "version_id": 1,
#     "endpoint_id": 312,
#     "account": "giza1"
#   },
#   "created_date": "2024-06-07T20:55:48.197225",
#   "last_update": "2024-06-07T20:55:48.197225"
# }

# giza agents create --model-id 739 --version-id 1 --endpoint-id 314 --name BETF6 --description BETF6
# 
# $ giza agents create --model-id 739 --version-id 1 --endpoint-id 314 --name BETF6 --description BETF6
# [giza][2024-06-07 14:18:48.002] Creating agent ✅
# [giza][2024-06-07 14:18:48.010] Using endpoint id to create agent, retrieving model id and version id
# [giza][2024-06-07 14:18:48.557] Select an existing account to create the agent.
# [giza][2024-06-07 14:18:48.561] Available accounts are:
# ┌──────────┐
# │ Accounts │
# ├──────────┤
# │  giza1   │
# └──────────┘
# Enter the account name: giza1
# {
#   "id": 74,
#   "name": "BETF6",
#   "description": "BETF6",
#   "parameters": {
#     "model_id": 739,
#     "version_id": 1,
#     "endpoint_id": 314,
#     "account": "giza1"
#   },
#   "created_date": "2024-06-07T21:18:52.209538",
#   "last_update": "2024-06-07T21:18:52.209538"
# }

# In[ ]:


get_ipython().system('giza users me')


# In[ ]:





# ## Run a verifiable inference
# ##### To streamline verifiable inference, you might consider using the endpoint URL obtained after transpilation. However, this approach requires manual serialization of the input for the Cairo program and handling the deserialization process. To make this process more user-friendly and keep you within a Python environment, we've introduced a Python SDK designed to facilitate the creation of ML workflows and execution of verifiable predictions. When you initiate a prediction, our system automatically retrieves the endpoint URL you deployed earlier, converts your input into Cairo-compatible format, executes the prediction, and then converts the output back into a numpy object. 

# In[8]:


from giza.agents.model import GizaModel

MODEL_ID = 739  # Update with your model ID
VERSION_ID = 1  # Update with your version ID

def prediction(input, model_id, version_id):
    model = GizaModel(id=model_id, version=version_id)

    (result, proof_id) = model.predict(
        input_feed={'input': input}, verifiable=True
    )

    return result, proof_id

def execution():
    # The input data type should match the model's expected input

    inputTotalETF = 453.1;

    input = np.array([[inputTotalETF]]).astype(np.float32)
    #input = np.array([[31,28.1]]).astype(np.float32)
    (result, proof_id) = prediction(input, MODEL_ID, VERSION_ID)

    print(
        f"Predicted value for input {input.flatten()[0]} is {result[0].flatten()[0]}")
    print(f"Proof ID: {proof_id}")

    return result, proof_id


execution()


# 🚀 Starting deserialization process...
# ✅ Deserialization completed! 🎉
# Predicted value for input 453.1000061035156 is 60012.989669799805
# Proof ID: 2ccaab8cc19642de988fd7e37f16f3e0
# (array([[60012.9896698]]), '2ccaab8cc19642de988fd7e37f16f3e0')

# ## Get and Download the proof
# #### Initiating a verifiable inference sets off a proving job on our server, sparing you the complexities of installing and configuring the prover yourself. Upon completion, you can download your proof.
# 
# First, let's check the status of the proving job to ensure that it has been completed.

# $ giza endpoints get-proof --endpoint-id 314 --proof-id "2ccaab8cc19642de988fd7e37f16f3e0"
# [giza][2024-06-07 14:11:49.529] Getting proof from endpoint 314 ✅
# {
#   "id": 1079,
#   "job_id": 1252,
#   "metrics": {
#     "proving_time": 19.864916
#   },
#   "created_date": "2024-06-07T21:11:22.794625"
# }

# Once the proof is ready, you can download it.

# $ giza endpoints download-proof --endpoint-id 314 --proof-id "2ccaab8cc19642de988fd7e37f16f3e0" --output-path giza-BETF-short-LIN-wip.proof
# 
# 
# $ giza endpoints download-proof --endpoint-id 314 --proof-id "2ccaab8cc19642de988fd7e37f16f3e0" --output-path giza-BETF-short-LIN-wip.proof
# [giza][2024-06-07 14:13:22.350] Getting proof from endpoint 314 ✅
# [giza][2024-06-07 14:13:25.223] Proof downloaded to giza-BETF-short-LIN-wip.proof ✅
# 

# ## Verify the proof
# #### Finally, you can verify the proof.

# $ giza verify --proof-id 1079
# 
# $ giza verify --proof-id 1079
# [giza][2024-06-07 14:14:02.287] Verifying proof...
# [giza][2024-06-07 14:14:03.884] Verification result: True
# [giza][2024-06-07 14:14:03.888] Verification time: 0.443269482
# 

# In[ ]:




