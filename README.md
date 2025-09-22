# Pocket-Rokets
***How It Works
This AI agent is designed to assist with task planning by taking a natural language goal, breaking it into actionable steps, enriching those steps with external data via web search and weather APIs, and generating a structured day-by-day or step-by-step plan. The plan is then saved in a SQLite database for later retrieval. A Streamlit web interface allows users to input new goals, view generated plans, and browse historical plans.***

Short Description:
1. Input: User provides a goal (e.g., "Plan a 2-day vegetarian food tour in Hyderabad").
2. Processing: A LangChain ReAct agent powered by Groq's qwen LLM reasons step-by-step, using tools to fetch web info (via Serper API) and weather data (via OpenWeather API).
3. Output: A formatted plan, stored in SQLite.
4. Interface: Streamlit app with pages for creating new plans and viewing history.



