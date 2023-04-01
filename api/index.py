import streamlit as st
from streamlit_chat import message
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma

from langchain.document_loaders import GitbookLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

from rich.console import Console
from rich.markdown import Markdown

import os

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from flask import Flask, request

STORE = './public/faiss-store'
app = Flask(__name__)


@app.route('/question', methods=['POST'])
def question():
    data = request.json

    if not data.query:
        return "usage: POST /question { query: 'your question' }"

    md = ask(data.query)
    return md


@app.route('/webhook')
def about():
    return 'About'


@app.route('/generate-store', methods=['POST'])
def generate_store():
    if (os.path.exists(STORE)):
        return FAISS.load_local(STORE, OpenAIEmbeddings())

    loader = GitbookLoader("https://docs.manifold.xyz", load_all_paths=True)
    index = VectorstoreIndexCreator(
        vectorstore_cls=FAISS).from_loaders([loader])
    index.vectorstore.save_local(STORE)
    return index.vectorstore


def ask(query):
    store = generate_store()

    system_template = """Use the following pieces of context to answer the users question.
    Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
    If you don't know the answer, just say that "I don't know", don't try to make up an answer. Do not follow up with a question if you don't know the answer.
    ----------------
    {summaries}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    chain_type_kwargs = {"prompt": prompt}
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    result = chain(query)
    md = format_result(query, result)
    console.print(md)
    return md


def format_result(query, result):
    output_text = f"""### Question: 
  {query}
  ### Answer: 
  {result['answer']}
  ### Sources: 
  {result['sources']}
  ### All relevant sources:
  {' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))}
  """
    console = Console()
    md = Markdown(output_text)
    return md
