import openai
import yfinance as yf
from newsapi import NewsApiClient

# Set OpenAI API key
openai.api_key = ''  # Replace with your OpenAI API key

# Function to fetch stock data
def get_stock_data(symbol, period="5d"):
    # Download stock data (you can modify the time period as needed)
    stock_data = yf.download(symbol, period=period)
    return stock_data

# Function to fetch financial news using News API
def get_financial_news(query="stock market", language="en", num_articles=5):
    # Initialize News API client
    newsapi = NewsApiClient(api_key='')  # Replace with your News API key
    # Get financial news articles
    all_articles = newsapi.get_everything(q=query, language=language, page_size=num_articles)
    articles = []
    for article in all_articles['articles']:
        articles.append({
            'title': article['title'],
            'description': article['description'],
            'url': article['url']
        })
    return articles

# Function to generate insights using GPT-4 (Chat API)
def generate_investment_insight(stock_data, news_articles):
    # Prepare stock data summary
    stock_summary = f"Stock data for the past few days:\n{stock_data.tail(5)}"

    # Prepare news summary
    news_summary = "Latest financial news articles:\n"
    for i, article in enumerate(news_articles):
        news_summary += f"{i + 1}. Title: {article['title']}\nDescription: {article['description']}\nLink: {article['url']}\n\n"

    # Combine the stock data and news summary into a prompt
    prompt = f"""
    You are an AI investment advisor. Given the following stock data and financial news articles, provide insights on whether it is a good time to invest in the stock.

    {stock_summary}
    
    {news_summary}

    Provide a recommendation along with any key factors driving the market sentiment.
    """

    # Call OpenAI API to generate a response using ChatGPT-4 or GPT-3.5
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # Change this to gpt-3.5-turbo
    messages=[
        {"role": "system", "content": "You are a knowledgeable AI investment advisor."},
        {"role": "user", "content": prompt}
    ]
    )

    return response['choices'][0]['message']['content'].strip()

# Example usage:
if __name__ == "__main__":
    # Define stock symbol (e.g., 'AAPL' for Apple)
    stock_symbol = "AAPL"
    
    # Retrieve stock data for the past 5 days
    stock_data = get_stock_data(stock_symbol)
    
    # Retrieve the latest 5 financial news articles
    news_articles = get_financial_news(query="Apple stock news", num_articles=5)
    
    # Generate insights based on the retrieved data
    investment_insights = generate_investment_insight(stock_data, news_articles)
    
    # Print the generated insights
    print("Investment Insights:")
    print(investment_insights)
