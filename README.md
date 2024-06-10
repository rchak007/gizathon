![image-20240609200437349](./Images/image-20240609200437349.png)





# Integrating AI and Web3 with Giza

## Overview

This project showcases the innovative integration of machine learning (ML) models with blockchain technology to create verifiable predictions that trigger smart contract actions. Using Giza, we bridge the gap between AI and Web3 by training a machine learning model, converting it into a verifiable format, and deploying it to a cloud endpoint. We then create an agent to make predictions with the model and execute smart contracts based on these predictions, all within a Streamlit application for an interactive user experience.

## Objectives

1. **Train an ML Model**: Develop a linear regression model using Scikit-Learn to predict closing prices based on total ETF values.
2. **Transpile the Model**: Convert the trained model into a format that can be verified and deployed using Giza's CLI tools.
3. **Deploy the Model**: Set up a verifiable inference endpoint using Giza, allowing us to call the model for predictions and verify its authenticity.
4. **Create an Agent**: Use Giza to create an agent that makes predictions with the model and triggers smart contracts based on the results.
5. **Run Verifiable Inferences**: Perform predictions and verify that the correct model was used, ensuring trust and transparency in the results.
6. **Interactive Application**: Utilize Streamlit to create a user-friendly interface for interacting with the model and blockchain, displaying real-time data and results.

## Key Components

### 1. Model Training and Transpilation

We start by training a linear regression model on the `BETF-Final.csv` dataset, which contains ETF values and closing prices. The model is then transpiled into an ONNX format using the `skl2onnx` library, making it compatible with Giza's tools for verifiable deployment.

#### Predicting Bitcoin Prices Using Bitcoin ETFs Inflow

##### Overview

In this project, we leverage the inflows of various Bitcoin Exchange-Traded Funds (ETFs) to predict the future price of Bitcoin. The ETFs used in our analysis include:

- IBIT
- FBTC
- BITB
- ARKB
- BTCO
- EZBC
- BRRR
- HODL
- BTCW
- GBTC
- TotalETF

We use the aggregate inflow of these ETFs, represented as `TotalETF`, as a key metric to predict Bitcoin prices. This approach combines the benefits of ETF investment data with machine learning to generate accurate and verifiable price predictions.

##### Why Use Bitcoin ETF Inflows?

Bitcoin ETFs are investment vehicles that allow investors to gain exposure to Bitcoin without directly purchasing the cryptocurrency. The inflows into these ETFs reflect the collective sentiment and interest of investors towards Bitcoin. By analyzing these inflows, we can gauge market trends and investor behavior, which are critical for predicting future price movements. Here are some reasons why Bitcoin ETF inflows are valuable for price prediction:

1. **Market Sentiment Indicator**: The amount of money flowing into Bitcoin ETFs can serve as a proxy for overall market sentiment. High inflows indicate strong investor interest and positive sentiment towards Bitcoin, suggesting potential price increases.
2. **Institutional Investment**: ETFs are often used by institutional investors who have significant market influence. Tracking ETF inflows helps us understand institutional investment trends, which can impact Bitcoin prices significantly.
3. **Liquidity and Demand**: ETF inflows directly affect the liquidity and demand for Bitcoin. High inflows typically increase demand, leading to price appreciation, while outflows can signal reduced interest and potential price declines.
4. **Regulatory Confidence**: The success and popularity of Bitcoin ETFs can also reflect regulatory confidence in Bitcoin as an asset class. Positive regulatory news and approvals often lead to increased ETF inflows and subsequent price hikes.





### 2. Model Deployment and Agent Creation

Using Giza's CLI, we deploy the transpiled model to create a verifiable inference endpoint. This endpoint allows us to make predictions and verify that the responses come from the trained model. We then create an agent that interacts with this endpoint, making it possible to call the model and trigger smart contracts based on the predictions.

### 3. Interactive Streamlit Application

The core of our project is an interactive application built with Streamlit. This app enables users to:

- Input data for predictions and see real-time results.
- Fetch current cryptocurrency prices using the CoinGecko API.
- Adjust their portfolio risk levels through an intuitive slider interface.
- View their current portfolio holdings and the recommended adjustments based on the model's predictions.

### 4. Blockchain Interaction

Through the agent, we connect the predictions to smart contract actions on the Ethereum blockchain. The app can:

- Rebalance the portfolio by swapping ETH and USDC tokens based on the model's recommendations.
- Display transaction details and Etherscan URLs for transparency.

## How It Works

1. **Model Training**: We train a linear regression model on historical ETF data to predict closing prices.
2. **Transpilation and Deployment**: The trained model is converted to ONNX format and deployed to a Giza-managed endpoint, ensuring verifiability.
3. **Agent Setup**: We create an agent that interfaces with the deployed model and the Ethereum blockchain.
4. **Streamlit Application**: Users interact with the model and blockchain through a Streamlit app, which fetches real-time data, makes predictions, and triggers smart contract actions.

### Commands and Execution

- **Transpile the Model**: `giza transpile linear_regression-betf6.onnx --output-path verifiable_betf_lr6`
- **Deploy the Endpoint**: `giza endpoints deploy --model-id 739 --version-id 1`
- **Create the Agent**: `giza agents create --model-id 739 --version-id 1 --endpoint-id 314 --name BETF6 --description BETF6`
- **Run the Streamlit App**: `streamlit run gizaAgentTest-INS-LIN.py`

### Benefits and Future Work

By integrating AI and blockchain, we create a system that is not only intelligent but also transparent and trustworthy. Users can confidently rely on the predictions, knowing they are verifiable and tamper-proof. Future enhancements could include:

- Expanding the model to include more features and improve prediction accuracy.
- Integrating additional cryptocurrencies and financial instruments.
- Enhancing the user interface for a more seamless experience.

### Conclusion

This project demonstrates the powerful combination of AI and Web3 technologies. By leveraging Giza's tools and Streamlit, we create a robust system for making verifiable predictions and executing smart contracts, opening new possibilities for secure and intelligent financial applications.