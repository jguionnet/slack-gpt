
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("The OPENAI_API_KEY environment variable must be set.")

if not os.environ["SLACK_APP_TOKEN"]:
    raise Exception("The SLACK_APP_TOKEN environment variable must be set.")

if not os.environ["SLACK_BOT_TOKEN"]:
    raise Exception("The SLACK_BOT_TOKEN environment variable must be set.")

# doc chunking parameters
CHUNK_SIZE = 1768 
# chunk overlap while chunking?
CHUNK_OVERLAP = 40 

# For PromptHelper which repacl text chunk based on constraints
# Context window for the LLM
MAX_INPUT_SIZE = 4097 # for  text-davinci-003
# Number of outputs for the LLM
NUM_OUTPUT = 512
# Chunk overlap as a ratio of chunk size
CHUNK_OVERLAP_RATIO = 0.01
 
# From $$$ -> $ but also per quality output 
# text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001
# other model do not support the API used I think?
# See https://platform.openai.com/docs/models/model-endpoint-compatibility
MODEL_NAME='text-davinci-003'

PERSIST_DIR = "./storage/" 