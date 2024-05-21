from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import constant
import logging
import os
import sys
from typing import List

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    GPTTreeIndex, 
    SimpleDirectoryReader,
    GPTVectorStoreIndex, 
    GPTListIndex,
    LLMPredictor,
    download_loader,
    QuestionAnswerPrompt,
    Document,
    PromptHelper,
    ServiceContext,
    StorageContext, 
    load_index_from_storage,
    load_indices_from_storage, 
    load_graph_from_storage,
    ResponseSynthesizer,
    )
from llama_index.node_parser import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.retrievers import VectorIndexRetriever, ListIndexRetriever, ListIndexEmbeddingRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.evaluation import QueryResponseEvaluator
from llama_index.readers import BeautifulSoupWebReader, TrafilaturaWebReader, SimpleWebPageReader
from langchain import OpenAI
from pathlib import Path
#from gpt_index import download_loader
import argparse
import json

# Define a global variable to hold the index
query_engine = None

# Set up the Slack app with your app token and socket mode handler
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])

def get_engine():
    global query_engine
    if query_engine is None:
        # Load the index from disk and store it in the global variable
        query_engine = load_index()
    return query_engine


def load_index(index_type="vector", temperature=0.0, similarity_top_k=2, 
                         response_mode="compact", verbose=False):
    print(f"load_index it:{index_type}, temp:{temperature}, top_k:{similarity_top_k}, rm:{response_mode}")

    # Configure the prompt parameters
    prompt_helper = PromptHelper(context_window=constant.MAX_INPUT_SIZE,
                                num_output=constant.NUM_OUTPUT,
                                chunk_overlap_ratio=constant.CHUNK_OVERLAP_RATIO)   

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=temperature, 
                                            model_name=constant.MODEL_NAME, 
                                            max_tokens=constant.NUM_OUTPUT))
    # store the customization
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    storage_context = StorageContext.from_defaults(persist_dir=constant.PERSIST_DIR+"/slack") 

    if index_type == "vector":
        index = load_index_from_storage(storage_context, index_id=index_type + "_index")
        # configure retriever
        retriever = VectorIndexRetriever(
            index=index, 
            similarity_top_k=similarity_top_k
        )
    else:
        raise Exception("Index type not supported")

 
    # Customize the prompt query
    QA_PROMPT_TMPL = (
        "The context information is below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given the context information and not prior knowledge, "
        "Give your best guess at answering the question: {query_str} \n"
        "Add a CONFIDENCE Score between 0 and 1 your overall answer\n"
        "If you don't know the answer based on the context, just say you don't know\n"
    )   
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

    # configure response synthesizer
    response_synthesizer = ResponseSynthesizer.from_args(
        service_context=service_context,
        text_qa_template = QA_PROMPT,
        response_mode= response_mode,
        node_postprocessors=[
            SimilarityPostprocessor()
        ]
    )
    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    return query_engine

# Define a listener function for the app_mention event
@app.event("app_mention")
def handle_app_mention(event, say):
    print(event)
    # Check if "?" was mentioned in the message
    if "?" in event["text"]:
        result = app.client.reactions_add(
            channel=event["channel"],
            timestamp=event["ts"],
            name="eyes"
        )
        # Search in the index and return the response       
        query_engine = get_engine()
        reply = query_engine.query(event["text"])
        say(reply.response.lstrip("\n"), thread_ts=event["ts"])
        print("response:", reply.response)
   
@app.command("/qai")
def my_command_handler(ack, say, command):
    from datetime import datetime
    # Acknowledge the command
    ack()
    print("command: ", command)
    say("Processing the question:" + command["text"]) 

    say(
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Command execution started..."
                }
            }
        ],
        channel=command["channel_id"],
        thread_ts=command["event"]["ts"],  # Retrieve the timestamp from command["event"]["ts"]
        icon_emoji=":hourglass_flowing_sand:"
    )
    query_engine = get_engine()
    reply = query_engine.query(command["text"])
    print("response:", reply.response)

    # Respond to the user
    say(reply.response.lstrip("\n"))


# Start the socket mode handler to listen for events
if __name__ == "__main__":
    handler.start()