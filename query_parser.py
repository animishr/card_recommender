from dotenv import load_dotenv

from utils.logging_config import LOGGER
from utils.query_graph import FEATURE_EXTRACTOR_GRAPH

load_dotenv()



def retrieve(user_input):
    initial_state = {"user_input": user_input}
    final_state = FEATURE_EXTRACTOR_GRAPH.invoke(initial_state)

    docs = final_state["context"]
    return docs


query = "I am a 28 year old woman architect living in New Delhi. I spend mostly booking tickets from " \
    "Indigo and buying groceries from Big basket. I am looking for a lifetime free card which provides " \
    "Airport Lounge Access benefits."

import pprint

pprint.pprint(retrieve(query))