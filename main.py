 #main.py - Core application logic for the mental health assistant.
# This file contains the LangGraph agent, state management, and RAG setup.
# It is designed to be called by the FastAPI backend, separating logic from the web interface.

import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import glob
import pandas as pd
from dotenv import load_dotenv

# --- Step 0: Environment Setup ---
# Load environment variables (like your OpenAI API key) from a .env file
load_dotenv()

# Securely get the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file or environment variables.")
os.environ["OPENAI_API_KEY"] = api_key

# Create a sample text file for the Retrieval-Augmented Generation (RAG) system if it doesn't exist.
# This ensures the application has some data to work with on first run.
if not os.path.exists("mental_health_docs"):
    os.makedirs("mental_health_docs")
    with open("mental_health_docs/anxiety_tips.txt", "w") as f:
        f.write("""Title: Managing Anxiety
    Anxiety can feel overwhelming, but there are many small, manageable steps you can take to help.
    1. Deep Breathing: Practice slow, deep breaths. Inhale for 4 seconds, hold for 7, and exhale for 8. This simple exercise can calm your nervous system.
    2. Limit Caffeine: Caffeine can heighten feelings of anxiety. Try to reduce your intake of coffee and energy drinks.
    3. Regular Exercise: Physical activity is a powerful stress reliever. Even a short walk can make a big difference.
    4. Mindfulness: Pay attention to the present moment without judgment. Try guided meditations available online.
    5. Talk to someone: Share your feelings with a trusted friend or family member.
    """)
    print("Created sample RAG content.")

# --- Step 1: Data Preparation (RAG) ---
print("Preparing RAG data...")
try:
    # Load documents from the specified directory
    loader = DirectoryLoader("./mental_health_docs/", glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # Split documents into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and build the FAISS vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()
    print("FAISS vector store created successfully.")
except Exception as e:
    print(f"Error preparing RAG data: {e}")
    # Exit if RAG setup fails as it's critical for the app
    exit()


# --- Step 2: LangGraph Agent & State Definition ---

# Define the structure for the agent's state. This TypedDict ensures data consistency across nodes.
class AgentState(TypedDict):
    chat_history: list[BaseMessage]
    question_count: int
    answers: list
    report: str
    user_id: str
    retrieved_chunks: List
    profile_summary: str
    severity: str
    safety_flag: bool

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# In-memory data store for user data. In a production app, this would be a database.
user_data = {}

# The list of assessment questions
questions = [
    "In the last two weeks have you frequently felt down, depressed, or hopeless? (Yes/No)",
    "Have you lately lost interest or pleasure in doing things you used to enjoy? (Yes/No)",
    "Are you having trouble sleeping (too much or too little)? (Yes/No)",
    "Have you felt more irritable or angry than usual? (Yes/No)",
    "Have you been feeling anxious or worried more often than usual? (Yes/No)",
    "Have you noticed changes in appetite or weight? (Yes/No)",
    "Have you had difficulty concentrating at work/school or on tasks? (Yes/No)",
    "Have you withdrawn from friends, family, or social activities? (Yes/No)",
    "Have you experienced panic-like symptoms (fast heart, shortness of breath) recently? (Yes/No)",
    "Have you had any thoughts of harming yourself or ending your life? (Yes/No)"
]


# --- LangGraph Node Functions ---

def score_and_retrieve_node(state: AgentState):
    """This node scores the user's answers and retrieves relevant documents from the vector store."""
    answers = state.get("answers", [])

    # Identify which questions were answered with "yes"
    positive_answers = [questions[i] for i, ans in enumerate(answers) if ans.lower().strip().startswith("y")]
    score = len(positive_answers)

    # Determine severity and check for safety flags (self-harm question)
    severity = "low"
    safety_flag = False
    if answers and answers[-1].lower().strip().startswith("y"):
        severity = "high"
        safety_flag = True
    elif score >= 6:
        severity = "moderate"
    elif score >= 3:
        severity = "mild"

    # Create a query for the retriever based on the user's answers
    query_str = "mental health tips for " + ", ".join([q.split("?")[0] for q in positive_answers]) if positive_answers else "general mental wellness tips"
    retrieved_chunks = retriever.invoke(query_str)

    profile_summary = f"The user answered 'Yes' to {len(positive_answers)} out of {len(questions)} questions. The calculated severity is {severity}."

    return {
        "retrieved_chunks": retrieved_chunks,
        "profile_summary": profile_summary,
        "severity": severity,
        "safety_flag": safety_flag
    }

def synthesize_report_node(state: AgentState):
    """This node generates the final, synthesized report for the user."""
    context = "\n".join([c.page_content for c in state.get("retrieved_chunks", [])])

    # A detailed prompt template to guide the LLM in generating a safe and helpful report
    prompt_template = PromptTemplate(
        template="""
        You are a calm, compassionate mental health assistant. Your role is to provide supportive, non-medical advice based *only* on the provided context. Do NOT invent information or give medical diagnoses.

        User's Profile Summary: {profile_summary}
        Context from Trusted Resources:
        ---
        {context}
        ---
        Task: Based on the user's profile and the provided context, generate a gentle and actionable report. Structure it as follows:

        **Summary:**
        A 2-3 sentence, plain-language summary of the user's current state based on their answers.

        **Actionable Steps:**
        A list of 3-5 simple, concrete self-care steps the user can try. Frame these as suggestions, not commands (e.g., "You might consider..." instead of "You should...").

        **Important Note:**
        A concluding paragraph reminding the user that this is not medical advice and encouraging them to seek professional help if they are struggling.

        If the user indicated any risk of self-harm, YOU MUST include the following safety message verbatim at the end of the report:
        "**Important Safety Message:** It sounds like you are going through a very difficult time. Please know that help is available and you are not alone. For immediate support, please contact the National Suicide Prevention Lifeline at 988 or the Crisis Text Line by texting HOME to 741741."
        """,
        input_variables=["profile_summary", "context"]
    )
    chain = prompt_template | llm
    report_text = chain.invoke({"profile_summary": state.get("profile_summary", ""), "context": context}).content

    # Ensure the safety message is added if the flag is set
    if state.get("safety_flag") and "**Important Safety Message**" not in report_text:
        safety_message = "\n\n**Important Safety Message:** It sounds like you are going through a very difficult time. Please know that help is available and you are not alone. For immediate support, please contact the National Suicide Prevention Lifeline at 988 or the Crisis Text Line by texting HOME to 741741."
        report_text += safety_message

    # Save the report to our in-memory user data store
    user_id = state.get("user_id", "default_user")
    if user_id not in user_data:
        user_data[user_id] = {"mood_logs": [], "reports": []}
    user_data[user_id]["reports"].append(report_text)

    return {"report": report_text}


# --- LangGraph Workflow Definition ---
# This graph is now ONLY for report generation after all questions have been answered.
workflow = StateGraph(AgentState)

# Add the defined functions as nodes in the graph
workflow.add_node("score_and_retrieve", score_and_retrieve_node)
workflow.add_node("synthesize_report", synthesize_report_node)

# Define the graph's structure and flow
workflow.add_edge(START, "score_and_retrieve")
workflow.add_edge("score_and_retrieve", "synthesize_report")
workflow.add_edge("synthesize_report", END)

# Compile the graph into a runnable application
app = workflow.compile()


# --- Helper Functions for Backend ---

def start_new_assessment(user_id: str):
    """Initializes a new assessment state with the first question."""
    first_question = questions[0]
    initial_state = {
        "chat_history": [AIMessage(content=first_question)],
        "question_count": 1, # We've now asked one question
        "answers": [],
        "user_id": user_id,
        "report": "",
        "retrieved_chunks": [],
        "profile_summary": "",
        "severity": "",
        "safety_flag": False
    }
    return initial_state

def process_user_answer(user_id: str, answer: str, current_state: dict):
    """
    Processes a user's answer. If questions remain, it asks the next one.
    If all questions are answered, it triggers the report generation graph.
    """
    # Append the user's message and update the answers list
    current_state["chat_history"].append(HumanMessage(content=answer))
    current_state["answers"].append(answer)

    question_count = current_state["question_count"]

    if question_count < len(questions):
        # If there are more questions, get the next one
        next_question = questions[question_count]
        current_state["chat_history"].append(AIMessage(content=next_question))
        current_state["question_count"] += 1
        return current_state
    else:
        # All questions have been answered, so generate the report using the graph.
        print("All questions answered. Generating report...")
        report_state = app.invoke(current_state)
        return report_state

def get_dashboard_data(user_id: str):
    """Retrieves mood logs and the latest report for a user."""
    if user_id not in user_data:
        return {"mood_logs": [], "last_report": "No reports generated yet."}

    user_info = user_data[user_id]
    last_report = user_info["reports"][-1] if user_info["reports"] else "No reports generated yet."
    return {
        "mood_logs": user_info.get("mood_logs", []),
        "last_report": last_report
    }

def add_mood_log(user_id: str, mood_value: int, mood_note: str):
    """Adds a new mood log for a user."""
    import datetime
    today = datetime.date.today().isoformat()
    if user_id not in user_data:
        user_data[user_id] = {"mood_logs": [], "reports": []}
    
    log_entry = {"date": today, "mood": mood_value, "note": mood_note}
    user_data[user_id]["mood_logs"].append(log_entry)
    return log_entry