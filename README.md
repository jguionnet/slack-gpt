# Slack GPT

Repo for reading slack channel, indexing them using llama_index and then exposing query methods

## Set up a development environment:
- Ensure you have Python 3.9 or later installed (https://www.python.org/downloads/).
- Set up a virtual environment: `python -m venv venv` (or `python3 -m venv .venv` if using Python 3 on macOS/Linux).
- Activate the virtual environment: `source .venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows).
- Install the necessary packages: `make install_dep`
- To remove a virtual environment rm -rf <env_name>

## Set up Open AI
- Create an account on Open AI (https://beta.openai.com/)
- Create an API key (https://beta.openai.com/account/api-keys)
- `export OPENAI_API_KEY=<your key>`

`curl -H "Authorization: Bearer $SLACK_BOT_TOKEN" https://slack.com/api/conversations.list?limit=1\&pretty=1`

Needed to add the app to the channel (right click on the channel -> "view channel details", "integrations" tab and add the app)

usage
- indexing
`python3 slack_dump.py`
- querying
`python3 clibot.py`


## Training data

Training data are coming from slack channel and it is using the (llamahub slack reader)[https://llamahub.ai/l/slack]  

Training data is used only for the index creation. It is not used for the query. The query is using the output of the index creation files

## Code overview

## Run
- Create simple vector index: `make load` or `python3 IndexAndQuery.py -bs data` 
- Query a simple vector index `make query` or `python3 IndexAndQuery.py -lq data` 
- Query using the basic webapp `make webapp` or `python3 IndexAndQuery.py -wq data`and then launch a browser at http://localhost:5000
- Compare different index performance `python3 playground.py -l data`

## Potential Next Steps

#

source ~/.zshrc # setup path to pyhton from home brew
source .venv*/bin/activate # setup path to pyhton from venv*

python3 -m pip install -r requirements6.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

python3 IndexAndQuery.py -ls data -it vector -m embeddings
python3 IndexAndQuery.py -ps pdf -it vector -m embeddings

# youtube
python3 IndexAndQuery.py -yt yt -it vector -m embeddings