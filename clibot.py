import logging
import os
import sys
from typing import List

import constant

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

# step 2: load the document and query
def load_index_and_query(index_type, temperature=0.0, similarity_top_k=1, 
                         response_mode="default", verbose=False):
    print(f"load_index_and_query it:{index_type}, temp:{temperature}, top_k:{similarity_top_k}, rm:{response_mode}")

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
    elif index_type == "list":
        index = load_index_from_storage(storage_context, index_id=index_type + "_index")
        # configure retriever
        retriever = ListIndexEmbeddingRetriever(
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
    # define evaluator
    evaluator = QueryResponseEvaluator(service_context=service_context)

    GREEN = '\033[32m'
    BLUE = '\033[34m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    # Define a set of questions used to test the options
    questions = ['When is the training week end?',
                 "When and where is the happy hour?"
                 "Tell me about the kids tri",
                 "Any merch store?"
                ]


    # query the index with the predefined Questions and print the response
    for q in questions:
        print(RESET)
        print(GREEN)
        gresponse = query_engine.query(q)
        print(f'index: {index_type}, response_mode {response_mode}'   
            + "\n" + q + "\n\n" + BLUE + BOLD + "Response: " + str(gresponse)+"\n") 

        # Expensive
        # eval_result = evaluator.evaluate_source_nodes(query=q, response=gresponse)
        # print("Query Evaluation: " + str(eval_result))

        if verbose:
            gresponse_json = {
                "response.text": str(gresponse)[0:50],
                "sources": [{"node.get_text": str(x.node.get_text)[0:50], 
                            #  "similarity": round(x.similarity, 2),
                            "node.ref_doc_id": str(x.node.ref_doc_id),
                            "node.extra_info": str(x.node.extra_info),                        
                            #  "start": x.node_info['start'],
                            #  "end": x.node_info['end']
                            } for x in gresponse.source_nodes]
            }
            print(json.dumps(gresponse_json, indent=2))
        print(RESET)

    while True:
        print(RESET)
        text_input = input("> ")
        gresponse = query_engine.query(text_input)
        print(f'index: {index_type}, response_mode {response_mode}'   
            + "\n" + text_input + "\n\n" + BLUE + "Response: " + str(gresponse)+ "\n")      

        if verbose:
            gresponse_json = {
            "text": str(gresponse),
            "sources": [{"node.get_text": str(x.node.get_text)[0:50], 
                        #  "similarity": round(x.similarity, 2),
                        "node.ref_doc_id": str(x.node.ref_doc_id),
                        "node.extra_info": str(x.node.extra_info),
                        #  "start": x.node_info['start'],
                        #  "end": x.node_info['end']
                        } for x in gresponse.source_nodes]
            }
            print(json.dumps(gresponse_json, indent=2))


if __name__ == "__main__":

    # define PoC CLI options
    parser = argparse.ArgumentParser()
    #     """Response modes."""
    # REFINE = "refine"
    # COMPACT = "compact"
    # SIMPLE_SUMMARIZE = "simple_summarize"
    # TREE_SUMMARIZE = "tree_summarize"
    # GENERATION = "generation"
    # NO_TEXT = "no_text"
    parser.add_argument('-rm', '--response_mode', type=str, help='Response mode (default "compact") other options see above')
    # tree summary: spit out all the header table in md format
    parser.add_argument('-tk', '--similarity_top_k', type=int, help='similarity_top_k (default 2)')
    parser.add_argument('-ts', '--txt_and_save', type=str, help='Load data in the index from specified directory')
    parser.add_argument('-lq', '--load_and_query', type=str, help='Load data from the index and query')
    parser.add_argument('-it', '--index_type', type=str, help='Type of index to generate or use (default "vector") other option "tree"')
    parser.add_argument('-t', '--temp', type=str, help='temperature')
    args = parser.parse_args()

    query = None
    tk = 2
    response_mode = "compact"
    index_type = "vector"
    temp=0
    txt = None

    if args.txt_and_save != None: txt = args.txt_and_save
    if args.load_and_query != None: query = args.load_and_query
    if args.similarity_top_k != None: tk = args.similarity_top_k
    if args.response_mode != None: response_mode = args.response_mode
    if args.index_type != None: index_type = args.index_type
    print("Running with Args: " + str(args))   

    load_index_and_query(index_type, temperature = temp, similarity_top_k=tk, 
                             response_mode=response_mode, verbose=False)   
   
