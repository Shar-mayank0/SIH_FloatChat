# core/query_engine.py
from core.db_setup import DB, get_engine
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from pydantic import SecretStr
import logging
from typing import Optional, Dict, Any, Tuple
from sqlalchemy import inspect, text
import re

class QueryEngine:
    def __init__(self, schema_metadata=None):
        print("\n[DEBUG] Initializing QueryEngine...")
        
        # Load environment variables first
        load_dotenv()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Get API key from environment
        groq_api_key_str = os.getenv('GROQ_API_KEY')
        if not groq_api_key_str:
            print("[ERROR] GROQ_API_KEY environment variable is not set")
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        print("[DEBUG] GROQ API key found")
        groq_api_key = SecretStr(groq_api_key_str)
        
        # Use the NeonDB engine from db_setup
        print("[DEBUG] Connecting to database...")
        self.engine = get_engine()
        self.db = SQLDatabase(self.engine)
        print("[DEBUG] Database connection established")
        
        print("[DEBUG] Initializing ChatGroq LLM...")
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        print("[DEBUG] LLM initialized")
        
        # Custom prompt template with schema information
        print("[DEBUG] Creating custom prompt template...")
        self.custom_prompt = self._create_custom_prompt()
        print("[DEBUG] Custom prompt created")
        
        # Override the output parser to clean SQL queries
        print("[DEBUG] Setting up SQLDatabaseChain...")
        self.db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm,
            db=self.db,
            verbose=True,
            use_query_checker=True,
            prompt=self.custom_prompt,
            return_direct=True  # Return SQL directly instead of executing it
        )
        print("[DEBUG] SQLDatabaseChain setup complete")
        self.schema = schema_metadata
        print("[DEBUG] QueryEngine initialization complete\n")
    
    def _create_custom_prompt(self) -> PromptTemplate:
        """Create a custom prompt template that includes detailed schema information."""
        print("[DEBUG] Building custom prompt template...")
        
        schema_info = """
        ARGO OCEANOGRAPHIC DATABASE SCHEMA:
        
        Table: argo_profiles
        Description: Main profiles table containing metadata for each Argo float profile
        Columns:
        - profile_id (INT, PRIMARY KEY): Unique identifier for each profile
        - platform_number (TEXT): Float platform identification number
        - project_name (TEXT): Name of the project/program
        - pi_name (TEXT): Principal Investigator name
        - cycle_number (INT): Cycle number of the float
        - direction (TEXT): Profile direction (A=ascending, D=descending)
        - data_centre (TEXT): Data center code
        - dc_reference (TEXT): Data center reference
        - data_state_indicator (TEXT): Data processing state
        - data_mode (TEXT): Real-time (R) or delayed-mode (D)
        - platform_type (TEXT): Type of platform/float
        - float_serial_no (TEXT): Serial number of the float
        - firmware_version (TEXT): Firmware version
        - wmo_instrument_type (TEXT): WMO instrument type code
        - juld (DOUBLE PRECISION): Julian date of the profile
        - juld_qc (TEXT): Quality control flag for julian date
        - juld_location (DOUBLE PRECISION): Julian date of location
        - latitude (DOUBLE PRECISION): Latitude in decimal degrees
        - longitude (DOUBLE PRECISION): Longitude in decimal degrees
        - position_qc (TEXT): Quality control flag for position
        - positioning_system (TEXT): Positioning system used
        - profile_pres_qc (TEXT): Overall QC flag for pressure
        - profile_temp_qc (TEXT): Overall QC flag for temperature
        - profile_psal_qc (TEXT): Overall QC flag for salinity
        - vertical_sampling_scheme (TEXT): Sampling scheme description
        - config_mission_number (INT): Mission configuration number
        - profile_date (DATE): Date of the profile
        - embedding (vector(1536)): Vector embedding for similarity search
        
        Table: measurements
        Description: Detailed measurements at different depth levels for each profile
        Columns:
        - id (SERIAL, PRIMARY KEY): Auto-incrementing unique identifier
        - profile_id (INT, FOREIGN KEY): Links to argo_profiles.profile_id
        - level (INT): Depth level number
        - pres (DOUBLE PRECISION): Pressure measurement (decibar)
        - pres_adjusted (DOUBLE PRECISION): Adjusted pressure measurement
        - temp (DOUBLE PRECISION): Temperature measurement (Celsius)
        - temp_adjusted (DOUBLE PRECISION): Adjusted temperature measurement
        - psal (DOUBLE PRECISION): Practical salinity measurement (PSU)
        - psal_adjusted (DOUBLE PRECISION): Adjusted salinity measurement
        - pres_adjusted_error (TEXT): Pressure adjustment error
        - temp_adjusted_error (TEXT): Temperature adjustment error
        - psal_adjusted_error (TEXT): Salinity adjustment error
        - pres_qc (TEXT): Quality control flag for pressure
        - pres_adjusted_qc (TEXT): QC flag for adjusted pressure
        - temp_qc (TEXT): Quality control flag for temperature
        - temp_adjusted_qc (TEXT): QC flag for adjusted temperature
        - psal_qc (TEXT): Quality control flag for salinity
        - psal_adjusted_qc (TEXT): QC flag for adjusted salinity
        
        
        RELATIONSHIP: measurements.profile_id → argo_profiles.profile_id (Many-to-One)
        
        IMPORTANT NOTES:
        1. Use adjusted values (_adjusted columns) when available for more accurate data
        2. Quality control flags: '1'=good, '2'=probably good, '3'=probably bad, '4'=bad, '9'=missing
        3. For temperature queries, use temp_adjusted when available, otherwise temp
        4. For pressure queries, use pres_adjusted when available, otherwise pres
        5. For salinity queries, use psal_adjusted when available, otherwise psal
        6. Always consider joining tables when you need both profile metadata and measurements
        7. Use LIMIT clause for large datasets to avoid performance issues
        8. Date formats: profile_date is DATE type, juld is Julian day format
        """
        
        template = f"""
        You are an expert SQL query generator for an Argo oceanographic database.
        
        {schema_info}
        
        Given an input question, create a syntactically correct PostgreSQL query to run.
        
        QUERY GUIDELINES:
        1. Always use table aliases for readability (ap for argo_profiles, m for measurements)
        2. When joining tables, use: measurements m JOIN argo_profiles ap ON m.profile_id = ap.profile_id
        3. For depth-related queries, use the 'level' column or pressure measurements
        4. For location queries, use latitude and longitude from argo_profiles
        5. For time-related queries, use profile_date or juld columns
        6. Always use LIMIT when the result might be large (default LIMIT 100)
        7. Use appropriate WHERE clauses to filter data efficiently
        8. For statistical queries, use appropriate aggregate functions (AVG, MAX, MIN, COUNT, etc.)
        
        CRITICAL FORMATTING REQUIREMENTS:
        - Return ONLY the SQL query with NO formatting, explanations, or markdown
        - Do NOT use ```sql, ```, or any code block markers
        - Do NOT include any text before the SQL query like "Here's the query:" or "SQL:"
        - Do NOT include any explanations after the SQL query
        - Start your response directly with SELECT, INSERT, UPDATE, DELETE, or WITH
        - Your entire response should be executable SQL code and nothing else
        
        EXAMPLES OF WHAT NOT TO DO:
        ❌ ```sql SELECT * FROM table;```
        ❌ Here's the SQL query: SELECT * FROM table;
        ❌ SELECT * FROM table; -- This query does...
        
        EXAMPLES OF CORRECT FORMAT:
        ✅ SELECT * FROM table;
        
        COMMON QUERY PATTERNS:
        - Profiles in a region: Use latitude/longitude bounds
        - Measurements at depth: Join tables and filter by pressure or level
        - Time series: Group by date or time periods
        - Data quality: Filter using QC flags
        - Float tracking: Use platform_number and cycle_number
        
        Only use the following tables: {{table_info}}
        
        Question: {{input}}
        
        Remember: Return ONLY the raw SQL query with no formatting or explanations."""
        
        print("[DEBUG] Custom prompt template built")
        return PromptTemplate(
            input_variables=["input", "table_info"],
            template=template
        )
    
    def send_groq_request(self, query: str):
        """Send a request to the Groq LLM and return the response."""
        print(f"[DEBUG] Sending query to Groq LLM: {query}")
        
        # Get table info from the database
        table_info = self.db.get_table_info()
        
        # Format the prompt with the query and table info
        formatted_prompt = self.custom_prompt.format(input=query, table_info=table_info)
        print(f"[DEBUG] Formatted prompt: {formatted_prompt}")
        
        # Send the formatted prompt to the LLM
        response = self.llm.invoke(formatted_prompt)
        print(f"[DEBUG] Received response from Groq LLM")
        print(f"[DEBUG] Raw LLM response: {response}")
        return response
        
    def get_sql_from_response(self, response) -> str:
        """Extract the SQL query from the LLM response."""
        print(f"[DEBUG] Extracting SQL from response")
        # Check if response has content attribute (LangChain response object)
        if hasattr(response, 'content'):
            sql_query = response.content.strip()
        else:
            # Fallback for string responses
            sql_query = str(response).strip()
        return sql_query
    
    # now execute the sql query
    def execute_sql(self, sql_query: str):
        """Execute the given SQL query and return the results."""
        print(f"[DEBUG] Executing SQL query: {sql_query}")
        with self.engine.connect() as connection:
            result = connection.execute(text(sql_query))
            rows = result.fetchall()
            print(f"[DEBUG] Query executed successfully, fetched {len(rows)} rows")
            return rows
    

