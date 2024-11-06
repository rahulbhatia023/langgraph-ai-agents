from typing import Union, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.tools import BaseTool
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import Field


class DocumentsRetrieverTool(BaseTool):
    pdf_file: str = Field(..., description="Uploaded PDF file")
    openai_api_key: str = Field(..., description="OpenAI API key")
    neo4j_uri: str = Field(..., description="Neo4j URI")
    neo4j_username: str = Field(..., description="Neo4j username")
    neo4j_password: str = Field(..., description="Neo4j password")

    name: str = "documents-retriever"
    description: str = "Retrieve similar documents chunks"

    def _run(self, query: str) -> Union[Dict, str]:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=self.openai_api_key,
            temperature=0,
        )

        documents = PyPDFLoader(self.pdf_file).load()

        graph_documents = LLMGraphTransformer(llm=llm).convert_to_graph_documents(
            documents
        )

        graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
        )

        graph.add_graph_documents(
            graph_documents, baseEntityLabel=True, include_source=True
        )

        vector_index = Neo4jVector.from_existing_graph(
            graph=graph,
            embedding=OpenAIEmbeddings(),
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )

        return "\n\n".join(
            doc.page_content for doc in vector_index.as_retriever().invoke(query)
        )
