# app.py - AI Agent for Task Planning using Langchain, Streamlit, Groq, Serper, OpenWeather

import os
import json
import sqlite3
from datetime import datetime

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
import requests

# Set up API keys from environment variables or Streamlit secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY") or os.getenv("SERPER_API_KEY")
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY") or os.getenv("OPENWEATHER_API_KEY")

if not all([GROQ_API_KEY, SERPER_API_KEY, OPENWEATHER_API_KEY]):
    st.error("Missing API keys. Please set GROQ_API_KEY, SERPER_API_KEY, and OPENWEATHER_API_KEY.")
    st.stop()

# Initialize LLM
llm = ChatGroq(temperature=0, model_name="qwen/qwen3-32b", groq_api_key=GROQ_API_KEY)

# Initialize Web Search Tool using Serper
search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
web_search_tool = GoogleSerperRun(api_wrapper=search)

# Custom Weather Tool using OpenWeather
@tool
def get_weather(location: str, date: str = None) -> str:
    """Get current weather or forecast for a location. If date is provided, attempt to get forecast; else current."""
    base_url = "https://api.openweathermap.org/data/2.5/"
    if date:
        # For forecast, use 5-day forecast API
        url = f"{base_url}forecast?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    else:
        url = f"{base_url}weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if date:
            # Simplistic: find forecast closest to date
            target_dt = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now()
            forecasts = data.get('list', [])
            closest = min(forecasts, key=lambda f: abs(datetime.fromtimestamp(f['dt']) - target_dt), default=None)
            if closest:
                return json.dumps({
                    "date": datetime.fromtimestamp(closest['dt']).strftime("%Y-%m-%d %H:%M"),
                    "temp": closest['main']['temp'],
                    "weather": closest['weather'][0]['description'],
                    "humidity": closest['main']['humidity']
                })
            else:
                return "No forecast available."
        else:
            return json.dumps({
                "temp": data['main']['temp'],
                "weather": data['weather'][0]['description'],
                "humidity": data['main']['humidity']
            })
    else:
        return f"Error fetching weather: {response.status_code}"

# Tools list
tools = [web_search_tool, get_weather]

# Prompt for the agent (adapted for ReAct)
prompt_template = """
You are a helpful task planning agent. Given a goal, break it into actionable steps, enrich with external info using tools, and output a clear, day-by-day or step-by-step plan.

You have access to the following tools:

{tools}

Use the following format:

Goal: the goal you must plan

Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final plan in a structured format, like:

Day 1:
- Step 1: ...
- Step 2: ...

Day 2:
- ...

Include relevant info from searches or weather.

Begin!

Goal: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Database setup
DB_FILE = "plans.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal TEXT NOT NULL,
            plan TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_plan(goal, plan):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO plans (goal, plan) VALUES (?, ?)", (goal, plan))
    conn.commit()
    conn.close()

def get_all_plans():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, goal, plan, created_at FROM plans ORDER BY created_at DESC")
    plans = cursor.fetchall()
    conn.close()
    return plans

# Streamlit App
st.title("AI Task Planner")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["New Plan", "View History"])

if page == "New Plan":
    goal = st.text_input("Enter your goal:", placeholder="e.g., Plan a 3-day trip to Jaipur with cultural highlights and good food")
    if st.button("Generate Plan"):
        if goal:
            with st.spinner("Generating plan..."):
                # Run agent
                response = agent_executor.invoke({"input": goal})
                plan = response['output']
                st.markdown("### Generated Plan")
                st.write(plan)
                save_plan(goal, plan)
                st.success("Plan saved!")
        else:
            st.error("Please enter a goal.")

elif page == "View History":
    plans = get_all_plans()
    if plans:
        for id, goal, plan, created_at in plans:
            with st.expander(f"Goal: {goal} (Created: {created_at})"):
                st.write(plan)
    else:
        st.info("No plans saved yet.")

# To run: streamlit run app.py