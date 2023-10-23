import llama_index
from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr 
import sys
import logging
import os
import json
import openai
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    Prompt,
    )

from llama_index.prompts import ChatPromptTemplate, ChatMessage, MessageRole
load_dotenv()  # take environment variables from .env.

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Set up logging configuration
logging.basicConfig(filename='Questions_asked.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def construct_index(directory_path):
    max_input_size = 4896
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 681 
    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio= 0.1, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    #INDEX
    index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

   #SAVE
    index.set_index_id("vector_index")
    index.storage_context.persist("./storage")

    return index


def chatbot(input_text):  
    
    #lagre_sporsmal(input_text)
    logging.info(f"Received question: {input_text}")

     # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    # load index
    index = load_index_from_storage(storage_context, index_id="vector_index")

    #PROMT

    #Create response
    
    TEMPLATE_STR = (
    "You are bot that only answers questions related to SATS in the language the question is phrased in, please try to create some linebreaks in your answers for readability, we have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n")
    
    QA_TEMPLATE = Prompt(TEMPLATE_STR)
    query_engine = index.as_query_engine(text_qa_template=QA_TEMPLATE, response_mode="compact")

    
    query_engine = index.as_query_engine(response_mode="compact")
    print("querying")
    response = query_engine.query(input_text)
    print("finished query")
    print("response",response)
    print("response.response",response.response)

    chain_str = "With the following context: " +
                "---------------------\n" +
                f"{response.response}" + "{context_str}" +
                "---------------------\n" +
    "Could you please create an api call to book "
    
    return response.response

    #This was to include where info came from: response = response.response + ' ' + json.dumps(response.metadata) 


index = construct_index("docs") 

iface = gr.Interface(fn=chatbot,
         inputs=gr.components.Textbox(lines=7, label="Skriv hva du lurer på, så svarer SATS AI"),
         outputs="text",
         title="SATS AI")
iface.launch(share=False)  # This line triggers the chat interface
