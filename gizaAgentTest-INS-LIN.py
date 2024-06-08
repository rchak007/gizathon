#!/usr/bin/env python
# coding: utf-8

# In[1]:
# conda env = giza3

# Locally run `streamlit run gizaAgentTest-INS-LIN.py`


import argparse
import logging
import os
import pprint
from logging import getLogger
import requests

import streamlit as st
# In[2]:


import numpy as np


# In[3]:


from ape import Contract, accounts, chain, networks


# In[4]:


from dotenv import find_dotenv, load_dotenv


# In[5]:


from giza.agents import AgentResult, GizaAgent


# In[6]:


from addresses import ADDRESSES


# In[7]:


# from lp_tools import get_tick_range
# from uni_helpers import (approve_token, check_allowance, close_position,
#                          get_all_user_positions, get_mint_params)


# In[8]:


load_dotenv(find_dotenv())


# In[9]:


dev_passphrase = os.environ.get("DEV_PASSPHRASE")
sepolia_rpc_url = os.environ.get("SEPOLIA_RPC_URL")
GIZA1_PASSPHRASE = os.environ.get("GIZA1_PASSPHRASE")


# In[ ]:





# In[10]:


logging.basicConfig(level=logging.INFO)


# In[11]:


def create_agent(
    model_id: int, version_id: int, chain: str, contracts: dict, account: str
):
    """
    Create a Giza agent for the regression model
    """
    agent = GizaAgent(
        contracts=contracts,
        id=model_id,
        version_id=version_id,
        chain=chain,
        account=account,
    )
    return agent


# In[12]:


def predict(agent: GizaAgent, X: np.ndarray):
    """
    Predict the next day volatility.

    Args:
        X (np.ndarray): Input to the model.

    Returns:
        int: Predicted value.
    """
    prediction = agent.predict(input_feed={"val": X}, verifiable=True, job_size="XL")
    return prediction


# In[13]:


def get_pred_val(prediction: AgentResult):
    """
    Get the value from the prediction.

    Args:
        prediction (dict): Prediction from the model.

    Returns:
        int: Predicted value.
    """
    # This will block the executon until the prediction has generated the proof
    # and the proof has been verified
    return prediction.value[0][0]


# In[14]:

def get_eth_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return data['ethereum']['usd']


# def eval_quote(
#     #agent_id: int,
#     pred_model_id:int,
#     pred_version_id:int,
#     policyId:int,
#     quote: float,
#     bmi:float,
#     account="chainaimlabs3003",
#     chain=f"ethereum:sepolia:{sepolia_rpc_url}",
# ):
    
#     ## Change the INS-ENROLLMENT-AGENT_PASSPHRASE to be {AGENT-NAME}_PASSPHRASE
#     # os.environ["INS_ENROLLMENT-AGENT_PASSPHRASE"] = os.environ.get("DEV_PASSPHRASE")

#     # Create logger

#     logger = getLogger("agent_logger")

#     networks.parse_network_choice(f"ethereum:sepolia:{sepolia_rpc_url}").__enter__()
#     chain_id = chain.chain_id

#     # Load the addresses
  
#     insuranceEnrollmentContractAddress = Contract(ADDRESSES["INSE"][chain_id])

#     #wallet_address = accounts.load(account).address

#     # Load the data, this can be changed to retrieve live data
#     #file_path = "data/data_array.npy"
#     #X = np.load(file_path)
     
#     #input = np.array([[30.1]]).astype(np.float32)

#     # input = np.array([[bmi]]).astype(np.float32)
#     inputTotalETF = 453.1;

#     input = np.array([[inputTotalETF]]).astype(np.float32)

#     # Fill this contracts dictionary with the contract addresses that our agent will interact with
#     contracts = {
#         "insuranceEnrollmentContract": insuranceEnrollmentContractAddress,
#     }

#     # Create the agent

#     agent = create_agent(
#         model_id=pred_model_id,
#         version_id=pred_version_id,
#         chain=chain,
#         contracts=contracts,
#         account=account,
#     )

#     result = predict(agent, input)

#     #if (quote<result){

#     #if(true)(

#     with agent.execute() as contracts:
        
#         logger.info("Executing INSE contract")
#         contracts.InsuranceEnrollmentContract.buyPolicy(policyId)
#         logger.info("Executing INSE contract")
#     #)

#     """
#     Get the value from the prediction.

#     Args:
#         prediction (dict): Prediction from the model.

#     Returns:
#         int: 

#     """
#     # This will block the executon until the prediction has generated the proof
#     # and the proof has been verified

#     print(result)

#     return result


# In[31]:


def  rebalance_lp(
    tokenWETH_amount: int,
    tokenUSDC_amount: int,
    pred_model_id: int,
    pred_version_id: int,
    account="giza1",
    chain1=f"ethereum:sepolia:{sepolia_rpc_url}",
    nft_id=None,
):
    logger = getLogger("agent_logger")
    networks.parse_network_choice(f"ethereum:sepolia:{sepolia_rpc_url}").__enter__()

    chain_id = chain.chain_id
    # weth_mint_amount = 0.01
    weth_mint_amount = tokenWETH_amount
    pool_fee = 3000
    uni = Contract(ADDRESSES["UNI"][chain_id])
    weth = Contract(ADDRESSES["WETH"][chain_id])
    # wbtc = Contract(ADDRESSES["WETH"][chain_id])
    wusdc = Contract(ADDRESSES["USDC"][chain_id])
    print("weth address: ", weth)
    # wbtc = Contract('0x66194f6c999b28965e0303a84cb8b797273b6b8b')
    weth_decimals = weth.decimals()
    # wbtc_decimals = wbtc.decimals()
    uni_decimals = uni.decimals()
    wusdc_decimals = wusdc.decimals()
    weth_mint_amount = int(weth_mint_amount * 10**weth_decimals)
    uni_mint_amount = int(0.5 * weth_mint_amount)
    contracts = {
        "weth": weth,
        "wusdc": wusdc,
    }
    # nft_manager_address = ADDRESSES["NonfungiblePositionManager"][11155111]
    # tokenA_address = ADDRESSES["UNI"][11155111]
    # tokenB_address = ADDRESSES["WETH"][11155111]
    # pool_address = "0x287B0e934ed0439E2a7b1d5F0FC25eA2c24b64f7"
    # print("pool address: ", pool_address)
    # contracts = {
    #     "nft_manager": nft_manager_address,
    #     "tokenA": tokenA_address,
    #     "tokenB": tokenB_address,
    #     "pool": pool_address,
    # }    


    inputTotalETF = 453.1;
    input = np.array([[inputTotalETF]]).astype(np.float32)

    # Create the agent

    agent = create_agent(
        model_id=pred_model_id,
        version_id=pred_version_id,
        chain=chain1,
        contracts=contracts,
        account=account,
    )

    result = predict(agent, input)
    print("model result = ", result)

    with agent.execute() as contracts:
    
        logger.info("Executing contract")
        # contracts.InsuranceEnrollmentContract.buyPolicy(policyId)
        # logger.info("Executing INSE contract")
        # print(f"Your WETH balance: {contracts.weth.balanceOf(wallet.address)/10**weth_decimals}")
        # print(f"Your WUSDC balance: {contracts.wusdc.balanceOf(wallet.address)/10**wusdc_decimals}")
    
    # with accounts.use_sender("giza1"):
    #     # print(f"Minting {weth_mint_amount/10**weth_decimals} WETH")
    #     # weth.deposit(value=weth_mint_amount)
    #     # print("Approving WETH for swap")
    #     # weth.approve(swap_router.address, weth_mint_amount)
    #     swap_params = {
    #         "tokenIn": weth.address,
    #         "tokenOut": wusdc.address,
    #         "fee": pool_fee,
    #         "recipient": wallet.address,
    #         "amountIn": weth_mint_amount,
    #         "amountOutMinimum": 0,
    #         "sqrtPriceLimitX96": 0,
    #     }
    #     swap_params = tuple(swap_params.values())
    #     print("Swapping WETH for USDC")
    #     # amountOut = swap_router.exactInputSingle(swap_params)
    #     # print(f"Successfully minted {uni_mint_amount/10**uni_decimals} USDC")
    

    


# In[37]:


def  rebalance_lp_NOMODEL(
    tokenWETH_amount: int,
    tokenUSDC_amount: int,
    pred_model_id: int,
    pred_version_id: int,
    account="giza1",
    chain1=f"ethereum:sepolia:{sepolia_rpc_url}",
    nft_id=None,
):
    logger = getLogger("agent_logger")
    networks.parse_network_choice(f"ethereum:sepolia:{sepolia_rpc_url}").__enter__()

    chain_id = chain.chain_id
    # weth_mint_amount = 0.01
    weth_mint_amount = tokenWETH_amount
    pool_fee = 3000
    uni = Contract(ADDRESSES["UNI"][chain_id])
    weth = Contract(ADDRESSES["WETH"][chain_id])
    # wbtc = Contract(ADDRESSES["WETH"][chain_id])
    wusdc = Contract(ADDRESSES["USDC"][chain_id])
    print("weth address: ", weth)
    # wbtc = Contract('0x66194f6c999b28965e0303a84cb8b797273b6b8b')
    weth_decimals = weth.decimals()
    # wbtc_decimals = wbtc.decimals()
    uni_decimals = uni.decimals()
    wusdc_decimals = wusdc.decimals()
    weth_mint_amount = int(weth_mint_amount * 10**weth_decimals)
    uni_mint_amount = int(0.5 * weth_mint_amount)
  
    wallet = accounts.load("giza1")

    # inputTotalETF = 453.1;
    # input = np.array([[inputTotalETF]]).astype(np.float32)

    # Create the agent

    # agent = create_agent(
    #     model_id=pred_model_id,
    #     version_id=pred_version_id,
    #     chain=chain1,
    #     contracts=contracts,
    #     account=account,
    # )

    # result = predict(agent, input)
    # print("model result = ", result)

    # with agent.execute() as contracts:
    
    logger.info("Executing contract")
    weth_bal = weth.balanceOf(wallet.address)/10**weth_decimals
    usdc_bal = wusdc.balanceOf(wallet.address)/10**wusdc_decimals
    
    print(f"Your WETH balance: {weth.balanceOf(wallet.address)/10**weth_decimals}")
    print(f"Your WUSDC balance: {wusdc.balanceOf(wallet.address)/10**wusdc_decimals}")
    return weth_bal, usdc_bal


# In[33]:


# MODEL_ID = 739
# VERSION_ID = 1

# print(MODEL_ID)
# print(VERSION_ID)

# tokenWETH_amount = 2500000
# tokenUSDC_amount = 1000000

# rebalance_lp(tokenWETH_amount, tokenUSDC_amount, MODEL_ID, VERSION_ID)


# ### No model call - just Ape contracts

# In[38]:


# MODEL_ID = 739
# VERSION_ID = 1

# print(MODEL_ID)
# print(VERSION_ID)

# tokenWETH_amount = 2500000
# tokenUSDC_amount = 1000000

# rebalance_lp_NOMODEL(tokenWETH_amount, tokenUSDC_amount, MODEL_ID, VERSION_ID)


# In[26]:

# Apply custom CSS
st.markdown("""
    <style>
    .stSlider > div > div > div > div {
        background-color: blue;
    }
    </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
     
    st.title("Giza ZKML model based Crypto Portfolio")
    
    MODEL_ID = 739
    VERSION_ID = 1

    print(MODEL_ID)
    print(VERSION_ID)

    # Display in the sidebar
    st.sidebar.header("Model Information")
    st.sidebar.write("Model ID:", MODEL_ID)
    st.sidebar.write("Version ID:", VERSION_ID)    

    # Fetch and display the current Ethereum price
    current_eth_price = get_eth_price()
    st.sidebar.markdown("---")
    st.sidebar.write("Current Ethereum Price:", current_eth_price)    

    # Slider for Risk Level
    st.sidebar.markdown("---")
    st.sidebar.header("Portfolio Risk Level")
    risk_level = st.sidebar.slider("Select Risk Level", min_value=1, max_value=10, value=5)


    tokenWETH_amount = 2500000
    tokenUSDC_amount = 1000000
    st.subheader("your current portfolio holdings")



    # rebalance_lp(tokenWETH_amount, tokenUSDC_amount, MODEL_ID, VERSION_ID)
    weth, usdc = rebalance_lp_NOMODEL(tokenWETH_amount, tokenUSDC_amount, MODEL_ID, VERSION_ID)

    wethUSD = weth * current_eth_price
    # Display the tokenWETH_amount
    st.metric("WETH", weth)
    st.write(f"WETH USD Value: {wethUSD:.2f}")
    st.metric("USDC Amount", usdc)


    # Add an extra line before the submit button
    st.text("")
    st.text("")

    if st.button("Call Agent"):
            commission = 0.001
            initialInvestment = 1000000
            interval='1d'

# In[ ]:




