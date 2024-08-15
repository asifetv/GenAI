import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


from langchain import SerpAPIWrapper
import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

from bs4 import BeautifulSoup as Soup
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="GlobeBotter", page_icon=":heart_eyes_cat:")
st.header(':cat: Welcome to Ezza, prrr. Muezza. ')
search = SerpAPIWrapper()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 200
)

import io
url = "https://ameenhousing.com"
loader = RecursiveUrlLoader(url=url, max_depth=1000, extractor=lambda x:Soup(x, "html.parser").text)
raw_documents = loader.load()
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings ())

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    output_key="output"
)

llm = ChatOpenAI()
tool = create_retriever_tool (
    db.as_retriever (),
    "Ameen Housing",
    "Searches and returns documents regarding Ameen Housing"
)

tools = [tool]

agent = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True)

user_query = st.text_input(
    "Prrr... How can I help you ?"
)

if "messages" not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you"}]
if "memory" not in st.session_state:
    st.session_state['memory'] = memory

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent(user_query)
        #response = agent(user_query, callback=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content":response})
        st.write(response)
    
if st.sidebar.button("Reset Chat History"):
    st.session_state_message = []