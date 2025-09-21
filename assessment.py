# assessment.py - FastAPI Backend
# This file creates the API endpoints that the frontend will interact with.
# It uses the logic from main.py to handle requests and serves the HTML file.

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uuid

# Import the core logic functions from our main.py file
from .main import start_new_assessment, process_user_answer, get_dashboard_data, add_mood_log

# Initialize the FastAPI application
app = FastAPI()

# In-memory session storage. For a real application, you'd use a database or Redis.
sessions = {}

# Tell FastAPI to look for HTML files in a directory named "templates"
templates = Jinja2Templates(directory="templates")

# Define the data models for our API requests to ensure type safety
class UserAnswer(BaseModel):
    session_id: str
    answer: str

class MoodLog(BaseModel):
    session_id: str
    mood_value: int
    mood_note: str


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main HTML file from the 'templates' directory.
    This is the entry point when a user visits the website.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start")
async def start_assessment():
    """
    Starts a new assessment session.
    It generates a unique session ID and gets the first question from the LangGraph agent.
    """
    # Create a unique ID for this user's session
    session_id = str(uuid.uuid4())
    user_id = f"user_{session_id}" # Associate a user ID with the session

    # Get the initial state and first question from the core logic
    initial_state = start_new_assessment(user_id)
    
    # Store the state in our session manager
    sessions[session_id] = initial_state
    
    # The first message is always the assistant's question
    first_question = initial_state['chat_history'][-1].content
    
    return {"session_id": session_id, "message": first_question}

@app.post("/chat")
async def chat_endpoint(user_answer: UserAnswer):
    """
    Processes the user's answer to a question.
    It retrieves the current session state, passes the answer to the LangGraph agent,
    and returns the agent's next response (either another question or the final report).
    """
    session_id = user_answer.session_id
    answer = user_answer.answer
    
    # Retrieve the current state for this session
    current_state = sessions.get(session_id)
    if not current_state:
        return {"error": "Invalid session ID"}, 404

    # Process the answer using the core logic
    new_state = process_user_answer(current_state['user_id'], answer, current_state)
    
    # Update the session with the new state
    sessions[session_id] = new_state
    
    # Determine the response type
    if new_state.get('report'):
        # If a report is generated, the assessment is over
        return {"type": "report", "message": new_state['report']}
    else:
        # Otherwise, send the next question
        next_question = new_state['chat_history'][-1].content
        return {"type": "question", "message": next_question}

@app.get("/dashboard/{session_id}")
async def get_user_dashboard(session_id: str):
    """
    Retrieves the dashboard data (mood logs and latest report) for a given session.
    """
    current_state = sessions.get(session_id)
    if not current_state:
        return {"error": "Invalid session ID"}, 404
        
    user_id = current_state['user_id']
    data = get_dashboard_data(user_id)
    return data

@app.post("/log_mood")
async def log_mood_endpoint(mood_log: MoodLog):
    """
    Logs a user's mood entry for a given session.
    """
    session_id = mood_log.session_id
    current_state = sessions.get(session_id)
    if not current_state:
        return {"error": "Invalid session ID"}, 404
        
    user_id = current_state['user_id']
    log_entry = add_mood_log(user_id, mood_log.mood_value, mood_log.mood_note)
    return {"status": "success", "log": log_entry}

# To run this FastAPI app:
# 1. Save the file as assessment.py
# 2. Open your terminal in the same directory.
# 3. Run the command: uvicorn assessment:app --reload

#cd c:\Users\pc\Desktop\mbbs_project
#uvicorn src.assessment:app --reload