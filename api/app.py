import pymupdf
from flask import Flask, jsonify, request
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient

MAX_CHUNK_SIZE = 1024
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small", chunk_size=MAX_CHUNK_SIZE
)
client = QdrantClient(url="http://localhost:6333")


@tool
def natural_language_query(input: str):
    """Will query the vector store that contains information about the research papers and return the top 10 most relevant chunks from all the papers.

    Args:
    input: A string that represents the query. This could be anything, a question, a sentence, etc. Feel free to try different queries if you are not able to get the desired results.

    Each result has the following structure:
    {
        "text": "string",
        "pdf_path": "string", # the path to the pdf file
        "page_number": 0, # the page number where the text is located
        "idx": string # the pdf is chunked into smaller pieces, this is the id of the chunk. eg: 1.txt, 2.txt, etc. Each document is split sequentially into chunks.
        "score": 0.0 # the cosine similarity between the input and the text
    }
    """
    result = client.search(
        collection_name="pdf_collection",
        query_vector=embeddings_model.embed_documents([input])[0],
        limit=10,
    )
    return [
        {
            "text": r.payload["text"],
            "pdf_path": r.payload["pdf_file"],
            "page_number": r.payload["page"],
            "idx": r.payload["idx"],
            "score": r.score,
        }
        for r in result
    ]


@tool
def read_pdf_page(pdf_file: str, page: int):
    """Will read the entire page for the pdf file and return the text.
    Args:
    pdf_file: The path to the pdf file.
    page: The page number to read from the pdf file.
    """

    pdf = pymupdf.open(pdf_file)
    return pdf[page - 1].get_text()


llm = ChatOpenAI(
    model="gpt-4o",
    max_retries=2,
    temperature=0.1,
)
agent_tools = [natural_language_query, read_pdf_page]
llm = llm.bind_tools(agent_tools)


research_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        You are a agent that is going to be searching technical research papers. You will be given a topic to find information on. You will need to find the most relevant information and return back the information with sources. 
        Make sure you only return information that you have retrieved from the research papers. Do not make up any information. You will have access to tools to help you find the information.

        Make sure to make uses of the tools provided. You can use the tools as many times as needed. Make sure you have all the information needed before returning the information. If you cannot find the information, just state that you could not find the information. Do not make up any information.

        Return the output by stating the information you found and the source of the information in the format pdf_path - page number - the section (if mentioned) in the papers we queried. The page number and pdf_path should only be from our vector store, do not make up any page numbers or pdf paths.
        Return only the bare minimum information needed and nothing more. Just enough to answer the question. Answer it concisely and only based on the information we have gotten through tool use.

        Always make sure to use read_pdf_page to make sure we have the right answer and understand the context of the information.
        Also make sure to use the natural_language_query tool many times with different queries that can be used to find the information needed. This means phrasing the question in different ways to get the most relevant information, asking your own questions, etc.

        eg: 
        // brief summary of the information only based on the tools we used
        // pdf_path 
        // page number
        // section (if it can be found in the text of the result)
        """,
        ),
        ("human", "Research {research_topic}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

app = Flask(__name__)


@app.route("/search", methods=["POST"])
async def handle_post():
    data = request.get_json()

    def agent_scratchpad_formatter(x):
        return format_to_openai_tool_messages(x["intermediate_steps"])

    agent = (
        {
            "research_topic": lambda x: x["research_topic"],
            "agent_scratchpad": agent_scratchpad_formatter,
        }
        | research_prompt
        | llm
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=agent_tools, verbose=True)
    research_agent_result = await agent_executor.ainvoke(
        {"research_topic": data['query']}
    )
    output = research_agent_result["output"]

    response = {"output": output}
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(debug=True)
