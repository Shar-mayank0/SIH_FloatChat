from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

class QueryEngine:
    def __init__(self, groq_api_key: str, schema_metadata=None):
        # Use the NeonDB engine from db_setup
        self.engine = get_engine() # type: ignore 
        self.db = SQLDatabase(self.engine)

        load_dotenv()
        groq_api_key = os.getenv('GROQ_API_KEY') # type: ignore 
        if groq_api_key is None:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.1-70b",   # or "mixtral-8x7b"
            temperature=0
        )

        self.db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm,
            db=self.db,
            verbose=True,
            use_query_checker=True
        )
        self.schema = schema_metadata

    def run_query(self, natural_language: str):
        return self.db_chain.run(natural_language)

    def get_schema(self):
        return self.db.get_table_info()