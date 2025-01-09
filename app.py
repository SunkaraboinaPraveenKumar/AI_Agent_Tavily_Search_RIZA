# Import necessary modules and setup for FastAPI, LangGraph, and LangChain
from fastapi import FastAPI  # FastAPI framework for creating the web application
from pydantic import BaseModel  # BaseModel for structured data data models
from typing import List  # List type hint for type annotations
from langchain_community.tools.tavily_search import TavilySearchResults
# TavilySearchResults tool for handling search results from Tavily
import os  # os module for environment variable handling
from langgraph.prebuilt import create_react_agent  # Function to create a ReAct agent
from langchain_groq import ChatGroq  # ChatGroq class for interacting with LLMs
import uvicorn  # Import Uvicorn server for running the FastAPI app
from langchain_community.tools.riza.command import ExecPython
from dotenv import load_dotenv
load_dotenv()

# Retrieve and set API keys for external tools and services
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["RIZA_API_KEY"] = os.getenv("RIZA_API_KEY")
# Predefined list of supported model names
MODEL_NAMES = [
    "llama3-70b-8192",
    "mixtral-8x7b-32768"
]

# Initialize the TavilySearchResults tool with a specified maximum number of results.
# Combine the TavilySearchResults and ExecPython tools into a list.
tools_tavily = TavilySearchResults(max_results=3)
tool_code_interpreter = ExecPython()
tools = [tools_tavily, tool_code_interpreter]

# FastAPI application setup with a title
app = FastAPI(title='LangGraph AI Agent')


# Define the request schema using Pydantic's BaseModel
class RequestState(BaseModel):
    system_prompt: str  # System prompt for initializing the model
    model_name: str  # Name of the model to use for processing the request
    messages: List[str]  # List of messages in the chat


# Define an endpoint for handling chat requests
@app.post("/chat")
def chat_endpoint(request: RequestState):
    try:
        if request.model_name not in MODEL_NAMES:
            return {"error": "Invalid model name. Please select a valid model."}

        # Initialize the LLM with the selected model
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=request.model_name)
        llm = llm.bind_tools(tools=tools)

        # Create the ReAct agent
        agent = create_react_agent(llm, tools=tools, state_modifier=request.system_prompt)

        # Prepare the state for the agent
        state = {"messages": request.messages}
        print(f"State before invoking agent: {state}")

        # Invoke the agent (ensure it’s awaited if needed)
        result = agent.invoke(state)
        print(f"Result after invoking agent: {result}")

        return result
    except Exception as e:
        return {"error": str(e)}  # Return the error message if any




# Run the application if executed as the main script
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)  # Start the app on localhost with port 8000
