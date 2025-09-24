# -Mental-Health-Assistant
Mental Health Assistant – A FastAPI + LangGraph web app for self-assessment, mood tracking, and personalized wellness reports using AI.

# 🧠 Mental Health Assistant

A web-based mental health self-assessment and mood tracker built with **FastAPI**, **LangGraph (LangChain)**, and a **Tailwind + Chart.js frontend**.
This project provides a safe, interactive space where users can answer self-reflection questions, generate a personalized report, and track their daily moods.

⚠️ **Disclaimer**: This is not a substitute for professional medical advice. If you are in crisis, please reach out to a licensed professional or call your local helpline immediately.


---
<img width="1514" height="1178" alt="Screenshot 2025-09-21 205214" src="https://github.com/user-attachments/assets/076279c0-7f93-4a15-8bf8-b5fc71841082" />

<img width="1669" height="1181" alt="Screenshot 2025-09-21 205534" src="https://github.com/user-attachments/assets/3a14bc25-d083-490c-bfab-5c7752b18c09" />





* 📋 **Self-Assessment Chat**
  Interactive 10-question “Yes/No” assessment with an AI assistant.

* 📑 **Personalized Report**
  Generates a supportive, non-medical report with actionable self-care suggestions.

* 📊 **Mood Tracker**
  Log daily moods (0–10 scale) with optional notes.

* 📈 **Dashboard**
  Visualize mood history using **Chart.js** and review your latest report.

* 🧩 **RAG (Retrieval-Augmented Generation)**
  Pulls contextual tips from a small knowledge base (`mental_health_docs/`) to enrich reports.

---

## 🛠️ Tech Stack

* **Backend**: FastAPI + Uvicorn
* **Core Logic**: LangGraph (built on LangChain) with OpenAI GPT models
* **Frontend**: HTML + TailwindCSS + Chart.js
* **Vector Store**: FAISS for RAG document retrieval
* **State Management**: In-memory sessions (replaceable with DB/Redis in production)

---

## 📂 Project Structure

```
├── assessment.py      # FastAPI backend with API routes
├── main.py            # Core logic (LangGraph agent, RAG, report synthesis)
├── index.html         # Frontend (assessment, tracker, dashboard UI)
├── mental_health_docs # Sample documents for RAG (created on first run)
└── requirements.txt   # Dependencies (create one if missing)
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/mental-health-assistant.git
cd mental-health-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

Make sure you have a `requirements.txt`. Suggested content:

```txt
fastapi
uvicorn
langchain
langgraph
langchain-openai
faiss-cpu
pandas
python-dotenv
jinja2
```

Then install:

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

Create a `.env` file in the root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the app

```bash
uvicorn assessment:app --reload
```



---

## 🖥️ Usage

1. Open the app in your browser.
2. Start an **Assessment** → answer 10 questions.
3. Receive a **Report**.
4. Log your daily mood in the **Tracker**.
5. View mood history + reports in the **Dashboard**.

---

## 🧩 Extending the Project

* Swap in a real database (Postgres, Mongo, Redis) for sessions and mood logs.
* Expand the RAG knowledge base (`mental_health_docs/`) with more wellness tips.
* Add authentication for multiple users.
* Deploy to cloud (e.g., Render, Railway, Heroku, or Docker + VPS).

---

## 📜 License

MIT License. Free to use, modify, and distribute.

---

