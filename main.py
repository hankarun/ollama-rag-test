from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

# Invoke chain with RAG context
llm = Ollama(model="codellama:7b")
## Load page content
loader = GenericLoader.from_filesystem(
    "/mnt/f/2_PROJECTS/simengine/libs",
    glob="**/*.cpp",
    parser=LanguageParser(
        parser_threshold=1000,
    ),
    show_progress=True
)
docs = loader.load()

## Vector store things
embeddings = OllamaEmbeddings(model="nomic-embed-text")

cpp_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.CPP, chunk_size=2000, chunk_overlap=200
)
texts = cpp_splitter.split_documents(docs)


vector_store = FAISS.from_documents(texts, embeddings)

## Prompt construction
prompt = ChatPromptTemplate.from_template(
    """
            Answer the following question only based on the given context
                                                    
            <context>
            {context}
            </context>
                                                    
            Question: {input}
"""
)

## Retrieve context from vector store
docs_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, docs_chain)

## Winner winner chicken dinner
response = retrieval_chain.invoke({"input": "Create an entity using scene and update?"})
print(":::ROUND 2:::")
print(response["answer"])