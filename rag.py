from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="codellama")
        self.text_splitter =  RecursiveCharacterTextSplitter.from_language(
            language=Language.CPP, chunk_size=2000, chunk_overlap=200
        )
        
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are senior c++ software developer. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )
        self.loader = GenericLoader.from_filesystem(
            "/mnt/f/2_PROJECTS/blueprints/simengine/include",
            glob="**/*.h",
            parser=LanguageParser(
                parser_threshold=1000,
            ),
            show_progress=True
        )
        self.docs = self.loader.load()
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        self.cpp_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.CPP, chunk_size=2000, chunk_overlap=200
        )
        self.texts = self.cpp_splitter.split_documents(self.docs)
        self.vector_store = FAISS.from_documents(self.texts, self.embeddings)        

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
        self.docs_chain = create_stuff_documents_chain(self.model, prompt)
        self.retriever = self.vector_store.as_retriever()
        self.chain = create_retrieval_chain(self.retriever, self.docs_chain)  

    def ask(self, query: str):
        return self.chain.invoke({"input": query})

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None