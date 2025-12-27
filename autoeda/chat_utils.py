import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

def init_agent(df, api_key):
    """
    Initializes the PandasAI agent (SmartDataframe) with OpenAI.
    
    Args:
        df: pandas DataFrame
        api_key: OpenAI API Key string
        
    Returns:
        SmartDataframe object or None if key is missing/invalid
    """
    if not api_key:
        return None
        
    llm = OpenAI(api_token=api_key)
    agent = SmartDataframe(df, config={"llm": llm})
    return agent

def chat_with_data(agent, query):
    """
    Sends a query to the agent and returns the response.
    
    Args:
        agent: SmartDataframe object
        query: str
        
    Returns:
        str (text response) or path/object (for plots)
    """
    if not agent:
        return "Agent not initialized. Please provide an API Key."
        
    try:
        response = agent.chat(query)
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"
