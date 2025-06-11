import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from openai import OpenAI
import os

# Read the API key from the file
try:
    with open("API_KEY", "r") as f:
        api_key = f.read().strip()
except FileNotFoundError:
    st.error("API_KEY file not found. Please create an API_KEY file with your OpenAI API key.")
    st.stop()

# Initialize the client
client = OpenAI(api_key=api_key)

def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        # Try multiple methods to get the price
        # Method 1: fast_info
        try:
            price = stock.fast_info.get("last_price")
            if price is not None:
                return f"Current price of {ticker}: ${price:.2f}"
        except:
            pass
        
        # Method 2: info
        try:
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if price is not None:
                return f"Current price of {ticker}: ${price:.2f}"
        except:
            pass
        
        # Method 3: history
        try:
            hist = stock.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                return f"Current price of {ticker}: ${price:.2f}"
        except:
            pass
            
        return f"Unable to retrieve price for {ticker}. Please check if the ticker symbol is correct."
        
    except Exception as e:
        return f"Error fetching stock price for {ticker}: {str(e)}"

def calculate_SMA(ticker, window):
    try:
        data = yf.Ticker(ticker).history(period='1y')
        close_prices = data['Close']

        if close_prices.empty:
            return f"Error: No price data returned for {ticker}"

        sma = close_prices.rolling(window=window).mean().iloc[-1]
        return str(sma)
    except Exception as e:
        return f"Error calculating SMA: {e}"

def calculate_EMA(ticker, window):
    try:
        data = yf.Ticker(ticker).history(period='1y')
        close_prices = data['Close']

        if close_prices.empty:
            return f"Error: No price data returned for {ticker}"

        ema = close_prices.ewm(span=window, adjust=False).mean().iloc[-1]
        return str(ema)
    except Exception as e:
        return f"Error calculating EMA: {e}"

def calculate_RSI(ticker):
    try:
        data = yf.Ticker(ticker).history(period='1y')
        close_prices = data['Close']

        if close_prices.empty:
            return f"Error: No price data returned for {ticker}"

        delta = close_prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        latest_rsi = 100 - (100 / (1 + rs)).iloc[-1]
        return str(latest_rsi)
    except Exception as e:
        return f"Error calculating RSI: {e}"

def calculate_MACD(ticker, window=0):  # Added default parameter
    try:
        data = yf.Ticker(ticker).history(period='1y')
        close_prices = data['Close']

        if close_prices.empty:
            return f"Error: No price data returned for {ticker}"

        short_EMA = close_prices.ewm(span=12, adjust=False).mean()
        long_EMA = close_prices.ewm(span=26, adjust=False).mean()
        MACD = short_EMA - long_EMA
        signal = MACD.ewm(span=9, adjust=False).mean()
        MACD_histogram = MACD - signal
        return f"MACD: {MACD.iloc[-1]:.2f}, Signal: {signal.iloc[-1]:.2f}, Histogram: {MACD_histogram.iloc[-1]:.2f}"
    except Exception as e:
        return f"Error calculating MACD: {e}"

def plot_stockprice(ticker):
    try:
        data = yf.Ticker(ticker).history(period='1y')
        if data.empty:
            return f"Error: No data available for {ticker}"
            
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['Close'])
        plt.title(f'{ticker} Stock Price Over the Last Year')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Use absolute path for better file handling
        filepath = os.path.abspath('stock.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return filepath  # Return full path instead of just filename
    except Exception as e:
        return f"Error plotting stock price: {e}"

def get_fundamentals(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        financials = ticker_obj.financials
        if financials.empty:
            return "No fundamental data available for this ticker"
        return financials.to_dict()
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
        if recs is None or recs.empty:
            return "No recommendations available."
        return recs.tail(5).to_dict()
    except Exception as e:
        return f"Error fetching recommendations: {e}"

def get_dividend_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "dividendYield": info.get("dividendYield", "N/A"),
            "dividendRate": info.get("dividendRate", "N/A"),
            "exDividendDate": info.get("exDividendDate", "N/A")
        }
    except Exception as e:
        return f"Error fetching dividend info: {e}"

def get_sentiment_summary(ticker):
    try:
        prompt = f"Analyze the sentiment and investment outlook for {ticker} based on its stock data , market sentiment , current news and fundamentals."
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        # Fixed: Access message content correctly
        return response.choices[0].message.content
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
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
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
        "description": "Ask GPT to generate a sentiment summary based on the company's data , stock data , market sentiment , current news and fundamentals.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    }
]

tool_defs = [{"type": "function", "function": f} for f in function_defs]

def call_gpt_with_function_call(user_query, tool_defs):
    try:
        # Step 1: Ask GPT with function call enabled
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": user_query}
            ],
            tools=tool_defs,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # Step 2: If function call was triggered
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # Call the corresponding function defined in your code
            result = globals()[func_name](**args)

            # Case 1: Result is a filepath (e.g., image)
            if isinstance(result, str) and result.endswith(".png"):
                if os.path.exists(result):
                    st.image(result, caption=f"Stock Chart for {args.get('ticker', 'Unknown')}")
                    st.success("Chart generated successfully!")
                else:
                    st.error("Failed to generate chart")
                return

            # Case 2: If result contains error, return it directly
            if isinstance(result, str) and ("Error" in result or "Unable" in result):
                return result

            # Case 3: Textual result, send it back to GPT for final response
            follow_up = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": user_query},
                    message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    }
                ],
                tools=tool_defs
            )
            return follow_up.choices[0].message.content

        else:
            # No function call needed; GPT responded directly
            return message.content
            
    except Exception as e:
        return f"Error processing request: {str(e)}"

# Streamlit UI
st.title("📊 Stock Market Analyst Assistant")
st.markdown("Ask me about any stock - I can provide prices, technical indicators, charts, and analysis!")

# Add some example queries
with st.expander("💡 Example Queries"):
    st.markdown("""
    - "What's the current price of AAPL?"
    - "Calculate the 50-day SMA for Tesla"
    - "Show me a chart of MSFT stock price"
    - "What's the RSI for GOOGL?"
    - "Get me fundamental data for AMZN"
    - "What are analysts saying about NVDA?"
    """)



query = st.text_input("Ask me about a stock:", placeholder="e.g., What's the current price of AAPL?")

if query:
    with st.spinner("Analyzing..."):
        result = call_gpt_with_function_call(query, tool_defs)
        if result:
            st.write(result)
with st.sidebar:
    st.header("📈 Tips")
    st.markdown("""
    **Supported Functions:**
    - Current stock prices
    - Technical indicators (SMA, EMA, RSI, MACD)
    - Stock charts
    - Company fundamentals
    - Analyst recommendations
    - Dividend information
    - Market sentiment analysis
    
    **Ticker Format:**
    Use standard ticker symbols like:
    - AAPL (Apple)
    - TSLA (Tesla)
    - MSFT (Microsoft)
    - GOOGL (Google)
    """)
