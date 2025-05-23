import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from openai import OpenAI

# Read the API key from the file
with open("API_KEY", "r") as f:
    api_key = f.read().strip()

# Initialize the client
client = OpenAI(api_key=api_key)

# Core Functions with Error Handling
def get_stock_price(ticker):
    try:
        return str(yf.Ticker(ticker).history(period='1y').iloc[-1].close)
    except Exception as e:
        return f"Error fetching stock price: {e}"

def calculate_SMA(ticker, window):
    try:
        data = yf.Ticker(ticker).history(period='1y').close
        return str(data.rolling(window=window).mean().iloc[-1])
    except Exception as e:
        return f"Error calculating SMA: {e}"

def calculate_EMA(ticker, window):
    try:
        data = yf.Ticker(ticker).history(period='1y').close
        return str(data.ewm(span=window, adjust=False).mean().iloc[-1])
    except Exception as e:
        return f"Error calculating EMA: {e}"

def calculate_RSI(ticker):
    try:
        data = yf.Ticker(ticker).history(period='1y').close
        delta = data.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        latest_rsi = 100 - (100 / (1 + rs)).iloc[-1]
        return str(latest_rsi)
    except Exception as e:
        return f"Error calculating RSI: {e}"

def calculate_MACD(ticker):
    try:
        data = yf.Ticker(ticker).history(period='1y').close
        short_EMA = data.ewm(span=12, adjust=False).mean()
        long_EMA = data.ewm(span=26, adjust=False).mean()
        MACD = short_EMA - long_EMA
        signal = MACD.ewm(span=9, adjust=False).mean()
        MACD_histogram = MACD - signal
        return f"MACD: {MACD[-1]:.2f}, Signal: {signal[-1]:.2f}, Histogram: {MACD_histogram[-1]:.2f}"
    except Exception as e:
        return f"Error calculating MACD: {e}"

def plot_stockprice(ticker):
    try:
        data = yf.Ticker(ticker).history(period='1y')
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['Close'])
        plt.title(f'{ticker} Stock Price Over the Last Year')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.grid(True)
        plt.savefig('stock.png')
        plt.close()
        return "Chart saved as stock.png"
    except Exception as e:
        return f"Error plotting stock price: {e}"

def get_fundamentals(ticker):
    try:
        return yf.Ticker(ticker).financials.to_dict()
    except Exception as e:
        return f"Error fetching fundamentals: {e}"

def get_volatility_and_beta(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "beta": info.get("beta", "N/A"),
            "52WeekHigh": info.get("fiftyTwoWeekHigh", "N/A"),
            "52WeekLow": info.get("fiftyTwoWeekLow", "N/A"),
            "volatility": info.get("regularMarketDayHigh", 0) - info.get("regularMarketDayLow", 0)
        }
    except Exception as e:
        return f"Error fetching volatility/beta: {e}"

def get_company_summary(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("longBusinessSummary", "Summary not available.")
    except Exception as e:
        return f"Error fetching summary: {e}"

def get_recommendations(ticker):
    try:
        recs = yf.Ticker(ticker).recommendations
        return recs.tail(5).to_dict() if recs is not None else "No recommendations available."
    except Exception as e:
        return f"Error fetching recommendations: {e}"

def get_dividend_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return {
            "dividendYield": stock.info.get("dividendYield", "N/A"),
            "dividendRate": stock.info.get("dividendRate", "N/A"),
            "exDividendDate": stock.info.get("exDividendDate", "N/A")
        }
    except Exception as e:
        return f"Error fetching dividend info: {e}"

def get_sentiment_summary(ticker):
    try:
        prompt = f"Analyze the sentiment and investment outlook for {ticker} based on its stock data and fundamentals."
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating sentiment summary: {e}"
    
function_defs = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL)"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "calculate_SMA",
        "description": "Calculate the Simple Moving Average for a stock over a specific window",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "window": {"type": "integer", "description": "Number of days for the moving average window"}
            },
            "required": ["ticker", "window"]
        }
    },
    {
        "name": "calculate_EMA",
        "description": "Calculate the Exponential Moving Average for a stock over a specific window",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "window": {"type": "integer", "description": "Number of days for the moving average window"}
            },
            "required": ["ticker", "window"]
        }
    },
    {
        "name": "calculate_RSI",
        "description": "Calculate the Relative Strength Index (RSI) for a given stock",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "calculate_MACD",
        "description": "Calculate the MACD indicator for a given stock",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "window": {
                    "type": "integer",
                    "description": "Unused but required by the function signature; can default to 0"
                }
            },
            "required": ["ticker", "window"]
        }
    },
    {
        "name": "plot_stockprice",
        "description": "Generate and save a stock price chart for the past year",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_fundamentals",
        "description": "Get financial statement data for a given stock",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_volatility_and_beta",
        "description": "Get volatility, beta, and 52-week high/low for a given stock",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_company_summary",
        "description": "Get a brief business description of the company",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_recommendations",
        "description": "Fetch recent analyst recommendations for a given stock",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_dividend_info",
        "description": "Get dividend information like yield and ex-dividend date",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_sentiment_summary",
        "description": "Ask GPT to generate a sentiment summary based on the company's data",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    }
]
def call_gpt_with_function_call(user_query, function_defs):
    # Step 1: Ask GPT with function call enabled
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": user_query}
        ],
        functions=function_defs,
        function_call="auto",
    )

    message = response.choices[0].message

    # Step 2: If function call was triggered
    if message.function_call:
        func_name = message.function_call.name
        args = json.loads(message.function_call.arguments)

        # Call the corresponding function defined in your code
        result = globals()[func_name](**args)

        # Case 1: Result is a filepath (e.g., image)
        if isinstance(result, str) and result.endswith(".png"):
            st.image(result, caption="Generated Plot")
            st.write("Here is the plot generated based on your request.")
            return

        # Case 2: Textual result, send it back to GPT for final response
        follow_up = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": user_query},
                message,
                {
                    "role": "function",
                    "name": func_name,
                    "content": str(result)
                }
            ]
        )
        st.write(follow_up.choices[0].message.content)

    else:
        # No function call needed; GPT responded directly
        st.write(message.content)
   



# Streamlit UI
st.title("📊 Stock Market Analyst Assistant")
query = st.text_input("Ask me about a stock:")
if query:
    st.write(call_gpt_with_function_call(query, function_defs))
