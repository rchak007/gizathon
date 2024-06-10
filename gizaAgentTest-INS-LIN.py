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
import re
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

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

# Declare global variables
chain_id = None
wethAddr = None
wusdcAddr = None
swap_router = None
weth_decimals = None
wusdc_decimals = None
pool_fee = 3000
wallet = None
dev_passphrase = os.environ.get("DEV_PASSPHRASE")
sepolia_rpc_url = os.environ.get("SEPOLIA_RPC_URL")
GIZA1_PASSPHRASE = os.environ.get("GIZA1_PASSPHRASE")
MODEL_ID = 739
VERSION_ID = 1

# In[ ]:





# In[10]:


logging.basicConfig(level=logging.INFO)


# In[11]:


def main():
    global chain_id, wethAddr, wusdcAddr, swap_router, weth_decimals, wusdc_decimals, pool_fee, wallet
    networks.parse_network_choice(f"ethereum:sepolia:{sepolia_rpc_url}").__enter__()
    chain_id = chain.chain_id
    
    wethAddr = Contract(ADDRESSES["WETH"][chain_id])
    wusdcAddr = Contract(ADDRESSES["USDC"][chain_id])
    swap_router = Contract(ADDRESSES["Router"][chain_id])
    weth_decimals = wethAddr.decimals()
    wusdc_decimals = wusdcAddr.decimals()
    pool_fee = 3000
    wallet = accounts.load("giza1")
    wallet.set_autosign(True, passphrase=dev_passphrase)


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
    # print("pred val = ", prediction.value)
    # print("type prediction value = ", type(prediction.value))
    # return prediction.value[0][0]
    try:
        # Debug: print the entire prediction object
        print("Full prediction object:", prediction)
        print("Type of prediction object:", type(prediction))

        # Debug: print the value attribute
        print("Prediction value attribute:", prediction.value)
        print("Type of prediction value attribute:", type(prediction.value))

        # Extract and return the predicted value
        if hasattr(prediction, 'value'):
            return prediction.value[0][0]
        else:
            print("Prediction object does not have a 'value' attribute.")
    except Exception as e:
        print(f"An error occurred while accessing prediction value: {e}")
        # raise
        st.session_state.inputNeeded = True

    return None  # Return None if the expected structure is not found

# In[14]:

def get_eth_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return data['ethereum']['usd']

def get_crypto_price(crypto_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_id}&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return data[crypto_id]['usd']





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
    # contracts = {
    #     "weth": weth,
    #     "wusdc": wusdc,
    # }
    wallet = accounts.load("giza1")
    wallet.set_autosign(True, passphrase=dev_passphrase)
    
    contracts = {
        "weth": "0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14",
        "wusdc": "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238",
    }   


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
        print(f"Your WETH balance: {contracts.weth.balanceOf(wallet.address)/10**weth_decimals}")
        # print(f"Your WUSDC balance: {contracts.wusdc.balanceOf(wallet.address)/10**wusdc_decimals}")

    return result
    
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



def balance_portfolio(total_value, current_eth_price, risk_level, btc_current_price, btc_predicted_price, scaling_factor=0.5):
    # Determine initial ETH allocation percentage based on risk level (1-10)
    initial_eth_allocation_percent = risk_level / 10
    
    # Calculate expected return based on BTC prices
    expected_return = (btc_predicted_price / btc_current_price) - 1
    
    # Adjust ETH allocation based on expected return and risk level
    adjusted_eth_allocation_percent = initial_eth_allocation_percent * (1 + scaling_factor * expected_return)
    
    # Ensure the allocation percentages are within bounds [0, 1]
    final_eth_allocation_percent = max(0, min(adjusted_eth_allocation_percent, 1))
    final_usdc_allocation_percent = 1 - final_eth_allocation_percent
    
    # Calculate the value to be allocated to ETH and USDC
    eth_value = total_value * final_eth_allocation_percent
    usdc_value = total_value * final_usdc_allocation_percent
    
    # Calculate the amount of ETH to hold
    eth_amount = eth_value / current_eth_price
    
    return eth_amount, usdc_value, final_eth_allocation_percent, final_usdc_allocation_percent


def perform_swap(current_eth_balance, current_usdc_balance, target_eth_amount, target_usdc_value, weth, wusdc, swap_router, pool_fee, wallet):
    # Calculate the differences
    eth_diff = target_eth_amount - current_eth_balance
    usdc_diff = target_usdc_value - current_usdc_balance
    print("target_eth_amount  = ", target_eth_amount)
    print("current_eth_balance = ", current_eth_balance)
    print("target_usdc_value = ", target_usdc_value)
    print("current_usdc_balance= ", current_usdc_balance)
    print("eth diff = ", eth_diff)
    st.metric("eth diff in Wei", (eth_diff * 10**weth.decimals()))
    st.metric("USDC diff in Wei", (usdc_diff * 10**wusdc.decimals()))
    print("usdc diff = ", usdc_diff)
    amountOut = 0

    try: 
        with accounts.use_sender("giza1"):
            if eth_diff > 0:
                # Need more ETH, swap USDC for ETH
                print(f"Swapping {eth_diff * 10**wusdc.decimals()} USDC for {eth_diff} ETH")
                intUsdcDiff = int(abs(usdc_diff * (10 ** wusdc_decimals)))
                print("intUsdcDiff = ", intUsdcDiff)

                print("Approving WETH for swap")
                try:
                    wusdcAddr.approve(swap_router.address, intUsdcDiff)
                except Exception as e:
                    print(f"Caught an exception of type: {type(e).__name__}")
                
                    swap_params = {
                        "tokenIn": wusdc.address,
                        "tokenOut": weth.address,
                        "fee": pool_fee,
                        "recipient": wallet.address,
                        "amountIn": intUsdcDiff,
                        # "amountIn": usdc_diff * 10**wusdc.decimals(),
                        "amountOutMinimum": 0,
                        "sqrtPriceLimitX96": 0,
                    }
            else:
                # Need more USDC, swap ETH for USDC
                # print(f"Swapping {abs(eth_diff)} ETH for {abs(usdc_diff * 10**wusdc.decimals())} USDC")
                print(f"Swapping {abs(eth_diff)} ETH or in Wei {abs(eth_diff * 10**weth.decimals())} ")
                intEthDiff = int(abs(eth_diff * (10 ** weth_decimals)))
                print("intEthDiff = ", intEthDiff)

                    
                print("Approving WETH for swap")
                try:
                    wethAddr.approve(swap_router.address, intEthDiff)
                except Exception as e:
                    print(f"Caught an exception of type: {type(e).__name__}")
                
                    swap_params = {
                        "tokenIn": weth.address,
                        "tokenOut": wusdc.address,
                        "fee": pool_fee,
                        "recipient": wallet.address,
                        "amountIn": intEthDiff,
                        "amountOutMinimum": 0,
                        "sqrtPriceLimitX96": 0,
                    }
            
            swap_params = tuple(swap_params.values())
            print("swap params = ", swap_params)
            amountOut = swap_router.exactInputSingle(swap_params)
            # weth_mint_amount = 0.01 * 10**weth.decimals()
            # weth_mint_amount = int(0.01 * 10**weth_decimals)
            # weth.deposit(value=weth_mint_amount)
            # st.write("deposit WETH = ", weth_mint_amount)
            # amoutout = weth_mint_amount
            # print(f"Swap completed. Output amount: {amountOut / 10**wusdc.decimals() if eth_diff > 0 else amountOut / 10**weth.decimals()}")
    except Exception as e:
        # Catch any exception, get its type, and print it
        print(f"Caught an exception of type: {type(e).__name__}")

    return amountOut

# Function to capture and return stdout and stderr output
def capture_output(func, *args, **kwargs):
    stdout = io.StringIO()
    stderr = io.StringIO()
    print("func is ", func)
    print("args are - ", *args)
    print("kwargs are - ", **kwargs)
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = func(*args, **kwargs)
    return result, stdout.getvalue(), stderr.getvalue()



# Extract Etherscan URLs using an alternative approach
def extract_etherscan_urls(text):
    urls = []
    lines = text.split('\n')
    for line in lines:
        if "https" in line:
            start_index = line.find("https")
            if start_index != -1 and len(line[start_index:]) == len("https://sepolia.etherscan.io/tx/0x3712e6309c08f08a491e5d1f9d6f48822b0e39b6d456357e459c8519c604e71c"):
                urls.append(line[start_index:])
    return urls


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

    # Initialize session state variables for prices
    if "current_eth_price" not in st.session_state:
        st.session_state.current_eth_price = get_crypto_price('ethereum')
    
    if "btc_price" not in st.session_state:
        st.session_state.btc_price = get_crypto_price('bitcoin')
    
    # Initialize session state variables for balance results and prediction value
    if "balance_results" not in st.session_state:
        st.session_state.balance_results = None

    # Initialize session state variables for input
    if "input_done" not in st.session_state:
        st.session_state.input_done = False
    
    if "prediction_value" not in st.session_state:
        st.session_state.prediction_value = None
    
    if "agent_predict_called" not in st.session_state:
        st.session_state.agent_predict_called = False

    if "agentResult" not in st.session_state:
        st.session_state.agentResult = False

    # Initialize session state variables
    if "inputNeeded" not in st.session_state:
        st.session_state.inputNeeded = False
    if "input_value" not in st.session_state:
        st.session_state.input_value = None
        
    if "balance_portfolio" not in st.session_state:
        st.session_state.balance_portfolio = False

    print("*************************")
    print("st.session_state.current_eth_price = ", st.session_state.current_eth_price)
    print("st.session_state.btc_price = ", st.session_state.btc_price)
    print("st.session_state.balance_results = ", st.session_state.balance_results)
    print("st.session_state.input_done = ", st.session_state.input_done)
    print("st.session_state.prediction_value = ", st.session_state.prediction_value)
    print("st.session_state.agent_predict_called = ", st.session_state.agent_predict_called)
    print("st.session_state.agentResult = ", st.session_state.agentResult)
    print("st.session_state.inputNeeded = ", st.session_state.inputNeeded)
    print("st.session_state.input_value = ", st.session_state.input_value)
    print("st.session_state.balance_portfolio = ", st.session_state.balance_portfolio)
    print("*************************")
    
    main()

     
    st.title("Giza ZKML model based Crypto Portfolio")
    


    print(MODEL_ID)
    print(VERSION_ID)

    # Display in the sidebar
    st.sidebar.header("Model Information")
    st.sidebar.write("Model ID:", MODEL_ID)
    st.sidebar.write("Version ID:", VERSION_ID)    


    # Initialize session state variables for prices
    if "current_eth_price" not in st.session_state:
        st.session_state.current_eth_price = get_crypto_price('ethereum')

    if "btc_price" not in st.session_state:
        st.session_state.btc_price = get_crypto_price('bitcoin')

    current_eth_price = st.session_state.current_eth_price
    btc_price = st.session_state.btc_price
    
    # Fetch and display the current Ethereum price
    # current_eth_price = get_eth_price()
    # Fetching the BTC price
    # btc_price = get_crypto_price('bitcoin')
    
    print(f"Current BTC Price: ${btc_price}")
    st.sidebar.markdown("---")
    st.sidebar.write("Current Ethereum Price:", current_eth_price)    
    st.sidebar.write("Current Bitcoin Price:", btc_price) 

    # Slider for Risk Level
    st.sidebar.markdown("---")
    st.sidebar.header("Portfolio Risk Level")
    risk_level = st.sidebar.slider("Select Risk Level", min_value=1, max_value=10, value=5)


    tokenWETH_amount = 2500000
    tokenUSDC_amount = 1000000
    st.subheader("your current portfolio holdings")



    # rebalance_lp(tokenWETH_amount, tokenUSDC_amount, MODEL_ID, VERSION_ID)
    # wethBal, usdcBal = rebalance_lp_NOMODEL(tokenWETH_amount, tokenUSDC_amount, MODEL_ID, VERSION_ID)
    wethBal = wethAddr.balanceOf(wallet.address)/10**weth_decimals
    usdcBal = wusdcAddr.balanceOf(wallet.address)/10**wusdc_decimals

    wethUSD = wethBal * current_eth_price
    # Display the tokenWETH_amount
    st.metric("WETH", wethBal)
    # st.metric("WETH Wei", wethBal * (10 ** weth_decimals))
    st.write(f"WETH USD Value: {wethUSD:.2f}")
    st.metric("USDC Amount", usdcBal)





    # Inputs
    # wallet = accounts.load("giza1")
    total_portfolio_value = wethUSD + usdcBal
    st.metric("Total current portfolio balance in USD ", total_portfolio_value)
    # current_eth_price = already computed before  # Current ETH price in USD
    btc_current_price = btc_price  # Current BTC price in USD -- 69401
    btc_predicted_price = 75000  # from Model prediction for future
    # risk_level = 7  # Risk level (1-10)

    # Using a scaling factor of 0.5
    scaling_factor = 0.5

    # Initialize session state variables
    if "balance_results" not in st.session_state:
        st.session_state.balance_results = None

    # Initialize session state variables
    if "prediction_value" not in st.session_state:
        st.session_state.prediction_value = None

    # Add an extra line before the submit button
    st.text("")
    st.text("")
    if st.session_state.agent_predict_called == False: 
        if st.button("Call Agent and give Rebalance Advise") and (st.session_state.agent_predict_called == False) and (st.session_state.prediction_value not in st.session_state):
                st.session_state.agent_predict_called = True
                commission = 0.001
                st.write("Your Risk Level you chose : ", risk_level)
                # result = rebalance_lp(tokenWETH_amount, tokenUSDC_amount, MODEL_ID, VERSION_ID)
                # st.write(f"result: {result}")
    
    
    #################   CALL MODEL to predict START
    
                chain=f"ethereum:sepolia:{sepolia_rpc_url}"
                account="giza1"
                contracts = {
                    "weth": "0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14",
                    "wusdc": "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238",
                }
            
                inputTotalETF = 453.1;
                input = np.array([[inputTotalETF]]).astype(np.float32)
            
                # Create the agent
                agent = create_agent(
                    model_id=MODEL_ID,
                    version_id=VERSION_ID,
                    chain=chain,
                    contracts=contracts,
                    account=account,
                )
    
                agentPredict = 0
                try:
                    result = predict(agent, input)
                    print("model result = ", result)
                    st.write("Model result = ", result)
                    print("type result = ", type(result))
                    agentPredict = 1
                    st.session_state.agentResult = True
                    
                except Exception as e:
                    # Catch any exception, get its type, and print it
                    print(f"Agent Predict function failed - Caught an exception of type - PREDICT FAILS: {type(e).__name__}")
                    st.write("Agent Predict function failed")
                    st.session_state.agentResult = False
                    if st.session_state.input_done is False:
                        st.session_state.inputNeeded = True
    
                predicted_value = None
                if agentPredict == 1:
                    try: 
                        st.session_state.prediction_value = get_pred_val(result)
                        predicted_value = st.session_state.prediction_value
                    except Exception as e:
                        # Catch any exception, get its type, and print it
                        print(f"get Value from Result fails - Caught an exception of type - PREDICT FAILS: {type(e).__name__}")
                        st.write("get Value from Result fails")
                        st.session_state.inputNeeded = True

                        if st.session_state.input_done is False:
                            st.session_state.inputNeeded = True
                            print("Failed to extract predicted value.")
                            st.write("Do Manual entry from result if result is there")
                            
                    
                        # Use session state to persist the predicted value input
                        if 'prediction_value' not in st.session_state:
                            st.session_state.prediction_value = None
    #################   CALL MODEL to predict END


    #################  Manual input section
    if (st.session_state.agentResult is True and st.session_state.prediction_value is None) or (st.session_state.agentResult is False):
        if (st.session_state.prediction_value is None) and  st.session_state.inputNeeded and (st.session_state.input_value is None) and (st.session_state.input_done is False):
            # Allow user to input the predicted value manually
            # st.session_state.predicted_value = st.number_input("Please input the predicted value manually from result:", min_value=0.0, format="%.5f")
            # st.session_state.predicted_value = 65000
            input_value = st.number_input( "Please input the predicted value manually from result:",
            min_value=0,  # Ensures only non-negative integers can be input
            step=1,  # Increment/decrement step size
            format="%d"  # Format as integer
            
            )
            if st.button("Submit Manual Value"):
                st.session_state.prediction_value = input_value
                st.session_state.input_value = input_value
                st.session_state.input_done = True
                st.session_state.inputNeeded = False
            # st.session_state.predicted_value = manual_val
            # print("st.session_state.predicted_value = ", st.session_state.predicted_value)
                if st.session_state.prediction_value:
                    st.write("Using manually inputted value for the next part.")
                    print("st.session_state.prediction_value - Using manually inputted value for the next part.", st.session_state.prediction_value)
                    predicted_value = st.session_state.prediction_value  # Use the user input for the next part
            
    #################  Manual input section END
        
            
    #################  Do the calculation for Balanced Portfolio
    # if st.session_state.prediction_value:
    if st.session_state.agent_predict_called and (st.session_state.prediction_value is not None) and st.session_state.prediction_value != 0:
            print("st.session_state.input_done = ", st.session_state.input_done)
            print("st.session_state.prediction_value = ", st.session_state.prediction_value)
        
            predicted_value = st.session_state.prediction_value
        
            btc_predicted_price = predicted_value
        
            # BTC Predicted Price = $75,000,  from MODEL
            if st.session_state.balance_portfolio == False:
                eth_amount_1, usdc_value_1, eth_allocation_1, usdc_allocation_1 = balance_portfolio(
                        total_portfolio_value, current_eth_price, risk_level, btc_current_price, btc_predicted_price, scaling_factor)
                
    
                # Store results in session state
                st.session_state.balance_results = {
                    "eth_amount_1": eth_amount_1,
                    "usdc_value_1": usdc_value_1,
                    "eth_allocation_1": eth_allocation_1,
                    "usdc_allocation_1": usdc_allocation_1
                }        
                st.session_state.balance_portfolio = True
        

            # Display results using Streamlit
            balance_results = st.session_state.balance_results
            usdc_value_1 = balance_results["usdc_value_1"]
            eth_amount_1 = balance_results["eth_amount_1"]
            eth_allocation_1 = balance_results["eth_allocation_1"]
            usdc_allocation_1 = balance_results["usdc_allocation_1"]
            st.write(f"BTC Predicted Price = {st.session_state.prediction_value}")
            st.write(f"Recommended ETH Amount: {eth_amount_1} ETH")
            st.write(f"Recommended USDC Value: ${usdc_value_1}")
            st.write(f"ETH Allocation: { eth_allocation_1 * 100:.2f}%")
            st.write(f"USDC Allocation: { usdc_allocation_1 * 100:.2f}%")

            # Now do the actual swap 
            # Perform the swap
    #################  Do the calculation for Balanced Portfolio _ END
    
    # Check if balance results are available in session state
    if ( st.session_state.balance_results is not None ) or st.session_state.balance_portfolio == True:            
        if st.button("Perform Rebalance"):
            balance_results = st.session_state.balance_results
            eth_amount_1 = balance_results["eth_amount_1"]
            usdc_value_1 = balance_results["usdc_value_1"]
            # Capture the output of the example_function and its return value
            # result, stdout_output, stderr_output = capture_output(example_function)
            
            result, stdout_output, stderr_output = capture_output(perform_swap,wethBal, usdcBal, eth_amount_1, usdc_value_1, wethAddr, wusdcAddr, swap_router, pool_fee, wallet)
            print("std out = ", stdout_output)
            print("std err ******* = ", stderr_output)
            print("result = ", result)

            # Extract Etherscan URLs using regex
            # etherscan_urls = re.findall(r'https://sepolia.etherscan.io/tx/\b[a-fA-F0-9]{64}\b', stdout_output)
            # Extract Etherscan URLs using regex from both stdout and stderr
            # etherscan_urls = re.findall(r'https://sepolia\.etherscan\.io/tx/\b[a-fA-F0-9]{64}\b', stdout_output + stderr_output)

            # Extract URLs from both stdout and stderr
            etherscan_urls = extract_etherscan_urls(stdout_output + stderr_output)            

            
            st.write("##### Etherscan URLs:")
            for url in etherscan_urls:
                st.write(url)
    
            # Display the captured output and the return value in Streamlit
            # st.write("### Captured stdout:")
            # st.write(stdout_output)
            # st.write("### Captured stderr:")
            # st.write(stderr_output)
            # st.write("### Result:")
            # st.write(result)

    
                # perform_swap(wethBal, usdcBal, eth_amount_1, usdc_value_1, wethAddr, wusdcAddr, swap_router, pool_fee, wallet)

# In[ ]:




