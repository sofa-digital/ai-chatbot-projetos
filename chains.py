import os
from dotenv import load_dotenv
import datetime
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import requests
from classes import ClassifyQuestion, FinalResponse, GlobalResponse, VendorIDResponse
from langchain_core.messages import ToolMessage
import json
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from services.Intranet_repository_s3 import IntranetRepository

load_dotenv()
# llm = ChatOpenAI(model="gpt-3.5-turbo")
llm = ChatOpenAI(model="gpt-4o-mini")

intranet_repository = IntranetRepository()
vectorstore = intranet_repository.create_or_load_faiss_index()

### classifier ###
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an advanced AI specialized in analyzing and extracting structured information from text.

            Current time: {time}

            Instructions:
            1. Analyze the user's input and classify it into one of the following categories:
            - 'vendorid': Queries related to specific title.            
            - 'global_question': General or unrelated queries.
            2. Identify and extract the 'vendor_id' if present. The vendor_id can appear in various forms, including but not limited to:
            - Phrases like "the vid is xxx ", "sku id is xxx", "vendor_id =s xxx", "ID: xxx", or any similar variation.
            - Formats such as alphanumeric (e.g., ABC123).
            3. Return the response in JSON format with the following structure:
            - 'request_type': The identified type of request.
            - 'vendor_id': The extracted vendor code, or null if none is found.
            
            Be flexible in recognizing variations of phrases and contexts, ensuring high accuracy in classification and code extraction..""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the last user's question above using the required format, and considering history when needed."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder = actor_prompt_template | llm.bind_tools(
    tools=[ClassifyQuestion], tool_choice="ClassifyQuestion"
)

### Final ###
def final_responder(input_messages):
    last_message = input_messages[-1]
    if hasattr(last_message, 'additional_kwargs') and \
                'tool_calls' in last_message.additional_kwargs:
        tool_calls = last_message.additional_kwargs['tool_calls']
        last_tool = tool_calls[-1]
        arguments = last_tool['function']['arguments']
        result = json.loads(arguments)
        tool_name = last_tool['function']['name']
        if tool_name == 'ClassifyQuestion':
            answer = result['request_type'].upper()
            if 'vendor_id' in result:
                answer += f" ({result['vendor_id']})"
        elif tool_name in ['GlobalResponse', 'VendorIDResponse']:
            answer = result['answer']
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        final_response = FinalResponse(answer=answer)
        return ToolMessage(
            name="FinalResponder",
            content=final_response.json(),
            tool_call_id=last_tool['id']
        )
    raise ValueError("No valid message found.")


### Global ###
def query_document(question, vectorstore, k=3):
    """
    Query the document repository with a question and return relevant contexts.
    Includes source information in the results.
    
    Args:
        question (str): The query string
        vectorstore: The FAISS vectorstore
        k (int): Number of results to return
        
    Returns:
        str: Concatenated context from relevant documents
    """
    docs = vectorstore.similarity_search(question, k=k)
    if docs:
        # Format the results to include source information
        results = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            results.append(f"[Source: {source}]\n{content}")
        
        return "\n\n".join(results)
    return "No relevant information found."

def build_prompt_with_context(question, context):
    prompt = f"""
    You are an expert assistant. Use the context below to answer the
      user's question accurately and concisely.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return prompt

def global_responder_logic(input_message):
    last_human_message = None
    for message in reversed(input_message):
        if isinstance(message, HumanMessage): 
            last_human_message = message.content
            break

    if not last_human_message:
        raise ValueError("No human message found in the input messages.")

    # Construir contexto e criar resposta
    context = query_document(last_human_message, vectorstore)
    prompt = build_prompt_with_context(last_human_message, context)
    response = llm.predict(prompt)
    global_response = GlobalResponse(answer=response)
    return global_response.json()

global_responder = global_responder_logic | llm.bind_tools(
    tools=[GlobalResponse], tool_choice="GlobalResponse"
)

### vendor_ID ###
def vendorid_responder_logic(input_message):
    if hasattr(input_message[-1], 'additional_kwargs') and \
        'tool_calls' in input_message[-1].additional_kwargs:
        tool_calls = input_message[-1].additional_kwargs['tool_calls']
        last_tool = tool_calls[-1]
        arguments = last_tool['function']['arguments']
        result = json.loads(arguments)
        vendor_id = result.get("vendor_id")
        if not vendor_id:
            raise ValueError("vendor_id not found.")
    else:
        raise ValueError("No valid message found to extract vendor_id.")
    url = f"http://vodcore.backend.sofadigital.com/get-title-by-vendor-id/{vendor_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        api_result = response.json()
        # {"id":"707a7776-e91b-4475-a8eb-f7852ba88b39","imdb_code":"0001_20120403_MOBZ_MEUPAIS","international_title":"Meu Pa\u00eds","molten_id":null,"original_title":"Meu Pa\u00eds","production_year":null,"release_date":null,"runtime":null,"vendor_id":"0001_20120403_MOBZ_MEUPAIS"}
        
        message = (
            f"Details for vendor_id {vendor_id}:\n"
            f"- IMDb Code: {api_result['imdb_code']}\n"
            f"- International Title: {api_result['international_title']}\n"
            f"- Original Title: {api_result['original_title']}\n"
            f"- Vendor ID: {api_result['vendor_id']}"
        )

        vendorid_response = VendorIDResponse(answer=message)
    except requests.RequestException as e:
        vendorid_response = VendorIDResponse(answer=f"Error: {str(e)}")
    return vendorid_response.json()

vid_responder = vendorid_responder_logic | llm.bind_tools(
    tools=[VendorIDResponse], tool_choice="VendorIDResponse"
)
