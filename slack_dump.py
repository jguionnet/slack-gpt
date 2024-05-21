# from https://llamahub.ai/l/slack
from llama_index import download_loader
import os
import logging
import sys
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

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("The OPENAI_API_KEY environment variable must be set.")

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
if not SLACK_BOT_TOKEN:
    raise Exception("The SLACK_BOT_TOKEN environment variable must be set.")

CHUNK_SIZE = 1768 
CHUNK_OVERLAP = 40 

# This loader fetches the text from a list of Slack channels. 
# You will need to initialize the loader with your Slack API Token or have the SLACK_BOT_TOKEN environment variable set
def slack_chunking(slack_channel_id, source):
    channel_ids = ['C02TTL455EK']

    txt_directory = source+"_txt"
    try:
        os.makedirs(txt_directory, exist_ok=True) 
    except OSError as error:
        print("Directory '%s' can not be created" % txt_directory)

    SlackReader = download_loader("SlackReader")

    loader = SlackReader()

    all_chunks = []
    chunk_count = 0
    index = 0
    # documents = loader.load_data(channel_ids=['[slack_channel_id1]', '[slack_channel_id2]'])
    # documents = loader.load_data(channel_ids=['C02TTL455EK'])
    documents = loader.load_data(channel_ids=channel_ids)
    text_splitter = TokenTextSplitter(separator=" ", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)    
    for document in documents:
        text_chunks = text_splitter.split_text(document.text)
        doc_chunks = [Document(t) for t in text_chunks]
        for doc_chunk in doc_chunks:
            chunk_count += 1
            doc_chunk.extra_info = {"chunk": str(chunk_count),
                                    }
            all_chunks.append(doc_chunk)
        filename = f"{txt_directory}/{slack_channel_id}.txt"
        with open(filename, "w") as f:
            f.write(document.text)
    index += 1
    return all_chunks

def load_parse_store_documents(all_chunks, source):

    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(all_chunks)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    # # initialize simple vector indices + global vector index
    # service_context = ServiceContext.from_defaults(chunk_size_limit=CHUNK_SIZE)
    #  Please provide a valid OpenAI model name.Known models are: gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, 
    # gpt-3.5-turbo, gpt-3.5-turbo-0301, text-ada-001, ada, text-babbage-001, 
    # babbage, text-curie-001, curie, davinci, text-davinci-003, text-davinci-002, 
    # code-davinci-002, code-davinci-001, code-cushman-002, code-cushman-001
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,chunk_size_limit=CHUNK_SIZE)

    vec_index = GPTVectorStoreIndex.from_documents(
        all_chunks, 
        service_context=service_context,
        storage_context=storage_context,
    )
    vec_index.set_index_id("vector_index")
    vec_index.storage_context.persist(persist_dir="./storage/" + source)

    list_index = GPTListIndex.from_documents(
        all_chunks,
        service_context=service_context,
        storage_context=storage_context)
    list_index.set_index_id("list_index")
    # save index to disk for further reuse
    list_index.storage_context.persist(persist_dir="./storage/" + source)


if __name__ == "__main__":

    # C02TTL455EK
    load_parse_store_documents(slack_chunking('general', 'slack') , 'slack')