from langchain.chains import RetrievalQAWithSourcesChain
from langchain.indexes import VectorstoreIndexCreator

from langchain.document_loaders import GitbookLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

from rich.console import Console
from rich.markdown import Markdown

import os
import re
import requests
import json

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from flask import Flask, request, jsonify

DISCOURSE_KEY = os.environ.get('DISCOURSE_KEY')
DISCOURSE_USERNAME = os.environ.get('DISCOURSE_USERNAME')
FORUM_URL = "https://forum.manifold.xyz"

STORE = './public/faiss-store'
app = Flask(__name__)


@app.route('/question', methods=['POST'])
def question():
    data = request.json

    if not data['query']:
        return "usage: POST /question { query: 'your question' }"

    answer = ask(data['query'])
    return jsonify({'answer': answer}), 200


@app.route("/webhook", methods=["POST"])
def discourse_webhook():
    payload = request.get_json(force=True)
    if payload["event_type"] == "topic_created":
        topic_id = payload["topic"]["id"]
        title = payload["topic"]["title"]
        content = payload["post"]["cooked"]

        # Remove HTML tags from the content
        content = re.sub(r'<[^>]*>', '', content)

        # Concatenate the title and content
        question = f"{title}\n{content}"

        # Remove any unsupported characters (retain ASCII characters and some common Unicode characters)
        question = re.sub(r'[^\x00-\x7F\u0080-\uFFFF]', '', question)

        answer = ask(question)
        create_post(topic_id, {'question': question, 'answer': answer})
        return jsonify({"status": "success"}), 200
    return jsonify({"status": "ignored"}), 200


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

    formatted = format_result(query, result)
    md = Markdown(formatted)
    console = Console()
    console.print(md)

    return formatted


def format_result(query, result):
    return f"""### Question: 
  {query}
  ### Answer: 
  {result['answer']}
  ### Sources: 
  {result['sources']}
  ### All relevant sources:
  {' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))}
  """


def create_post(topic_id, content):
    url = f"{FORUM_URL}/posts"
    headers = {
        "Content-Type": "application/json",
        "Api-Key": DISCOURSE_KEY,
        "Api-Username": DISCOURSE_USERNAME,
    }
    data = {
        "topic_id": 3632,  # topic_id,
        "raw": json.dumps(content, 2),
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()
