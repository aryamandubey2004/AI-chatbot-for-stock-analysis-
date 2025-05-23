# AI-chatbot-for-stock-analysis-
AI chatbot for in-depth stock analysis using SMA, EMA, RSI, sentiment analysis
# 📈 Stock Analysis Assistant Chatbot

A conversational AI assistant that provides stock market insights, technical indicators, fundamental data, and price visualizations using natural language queries. Powered by OpenAI's function calling  and built with Streamlit and `yfinance`.

---

## 🚀 Features

- ✅ Chat with the assistant to:
  - Get real-time stock prices
  - Analyze technical indicators: SMA, EMA, RSI, MACD
  - Generate sentiment summaries using GPT
  - View analyst recommendations
  - Access company fundamentals and dividend data
  - Visualize stock prices (charts)
- 🔧 Pluggable with:
  - **OpenAI API** (`gpt-4o`)
- 📊 Built using:
  - `Streamlit` for UI
  - `yfinance` for stock data
  - `matplotlib` for plotting
  - `openai` or `requests` for LLMs

---

## 📂 Project Structure

```bash
.
├── main.py              # Streamlit app with function calling logic
├── stock.png            # Output plot (generated dynamically)
└── README.md
