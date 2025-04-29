# Delta Logistic Intranet Assistant

## Description

The Delta Logistic Intranet Assistant is an AI-powered chatbot designed to streamline employee interactions with the company's intranet. It enables users to query general company information, check salary details, and view vacation balance, offering seamless integration with Delta Logistic's internal systems. Powered by LangChain, FAISS, and FastAPI, the assistant delivers intelligent responses by combining real-time data retrieval and a pre-built knowledge base.

Key Features:
- **Intelligent Classification**: Automatically classifies user input into salary queries, vacation balance requests, or general questions.
- **Real-time API Integration**: Fetches live data for payroll and leave balances via FastAPI endpoints.
- **Knowledge Retrieval**: Leverages a pre-built FAISS index to answer general questions based on intranet documentation.
- **Streamlit Interface**: Offers an intuitive chat-based interface for employees.

## Requirements

### System Requirements
- **Python**: Version 3.10 or higher

### Python Dependencies
- `python-dotenv`
- `langchain`
- `langchain-openai`
- `langgraph`
- `streamlit`
- `chromadb`
- `sentence-transformers`
- `faiss-cpu`
- `fastapi`
- `uvicorn`

Install the dependencies using `poetry` (defined in `poetry.toml`).

### Environment Variables
Configure the following variables in a `.env` file:
```plaintext
OPENAI_API_KEY=xxxx
LANGCHAIN_API_KEY=xxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=xxxxxxx
VACANCY_ENDPOINT_URL=http://127.0.0.1:8000/employee/vacancy
SALARY_ENDPOINT_URL=http://127.0.0.1:8000/employee/payroll
```

## Mode of Use
### Clone the Repository
git clone https://github.com/ferrerallan/ai-engage-agentic.git
cd ai-engage-agentic

### Install Dependencies
poetry install

### Run the Backend API
bash start_api.ch

### Build the FAISS Index
python -m services.intranet_repository

### Launch the Chat Application
bash start_chat.ch


## Usage Instructions
1. Open the chat interface in your browser.

2. Ask questions about the company, salary details, or vacation balance. Example inputs:
   - "How many vacation days do I have left? My code is abc123."
   - "Tell me about my YTD salary. Employee code: def456."
   - "What is the company's remote work policy?"

3. View responses directly in the chat interface.


## Implementation Details

### Backend API
- **Endpoints**:
  - `/employee/vacancy`: Retrieves vacation balance based on employee code.
  - `/employee/payroll`: Retrieves year-to-date payroll information.
- **Database**: Uses SQLite for storing employee and earnings data, initialized with sample records.

### Knowledge Retrieval
- **FAISS Index**: Built from intranet documentation (`docs/Delta_Logistic_intranet.txt`) using OpenAI embeddings.
- **Querying**: Matches user queries to relevant intranet content using semantic similarity.

### Conversational Flow
The chatbot uses a directed graph to route user inputs:
1. **Classifier**: Determines query type (`salary_request`, `vacancy_request`, `global_question`).
2. **Responders**:
   - **Salary Responder**: Calls the payroll API.
   - **Vacancy Responder**: Calls the vacation balance API.
   - **Global Responder**: Retrieves answers from the FAISS index.
3. **Final Responder**: Formats the response and displays it to the user.

### Core Files
- `backend/api.py`: FastAPI implementation for salary and vacation balance endpoints.
- `services/intranet_repository.py`: Manages FAISS index creation and document queries.
- `app.py`: Streamlit-based chatbot interface.
- `chains.py`: Defines responders and integrates APIs with the conversation graph.
- `classes.py`: Pydantic models for structured request and response handling.

