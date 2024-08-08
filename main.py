from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

## create graph
from pyvis.network import Network
from IPython.display import HTML, display

from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.core import PromptTemplate

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
h1b_data_path = os.path.join("data", "filtered_h1b_data.csv")
df = pd.read_csv(h1b_data_path)


instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python Pandas expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)

pandas_prompt_str = (
    "You are working with a pandas dataframe df.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "From the three possible solutions, choose the most accurate pandas query.\n"
    "Possible Solutions: {variations_output}\n"
    "Expression:"
)
response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)

query_variation_prompt_str = (
    "Given the following question, generate three pandas queries to answer the question.\n"
    "Each of the pandas queries should work with the dataframe df to answer the question.\n"
    "The result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Question: {query_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Pandas Queries 1.:\n"
)

query_variation_prompt = PromptTemplate(query_variation_prompt_str).partial_format(instruction_str=instruction_str, df_str=df.head(5))

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5)
)

pandas_output_parser = PandasInstructionParser(df)

response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
llm = OpenAI(model="gpt-3.5-turbo")

qp = QP(
    modules={
        "input": InputComponent(),
        "query_variation_prompt":query_variation_prompt,
        "llm1":llm,
        "pandas_prompt": pandas_prompt,
        "llm2": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm3": llm,
    },
    verbose=True,
)

qp.add_chain(["input", "query_variation_prompt","llm1"])
qp.add_link("input", "pandas_prompt",dest_key="query_str")
qp.add_link("llm1", "pandas_prompt",dest_key="variations_output")
qp.add_link("pandas_prompt","llm2")
qp.add_link("llm2","pandas_output_parser")
qp.add_link("pandas_output_parser","response_synthesis_prompt", dest_key="pandas_output")
qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
qp.add_link("llm2", "response_synthesis_prompt", dest_key="pandas_instructions")
qp.add_link("response_synthesis_prompt", "llm3")


net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(qp.dag)
net.save_graph("rag_dag.html")


while True:
    user_input = input("Enter your query: ")
    if user_input.lower() == 'q':
        break
    else:
        response = qp.run(query_str=user_input)
        print(response.message.content)






