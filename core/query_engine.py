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
    
    def _clean_sql_query(self, query_text: str) -> str:
        """
        Clean and extract SQL query from LLM response.
        Simple, direct approach to remove markdown and extract SQL.
        """
        print("\n[DEBUG] _clean_sql_query() - Raw input:")
        print(f"[DEBUG] {repr(query_text)}")
        
        if not query_text:
            print("[ERROR] Empty query text received")
            raise ValueError("Empty query text received")
        
        # Step 1: Remove the SQLQuery: prefix
        print("[DEBUG] Step 1: Removing SQLQuery: prefix")
        query_text = re.sub(r'^SQLQuery:\s*', '', query_text, flags=re.IGNORECASE)
        print(f"[DEBUG] After Step 1: {repr(query_text)}")
        
        # Step 2: Extract content between ```sql and ``` if present
        print("[DEBUG] Step 2: Extracting content between ```sql and ```")
        sql_block_match = re.search(r'```sql\s*(.*?)\s*```', query_text, re.DOTALL | re.IGNORECASE)
        if sql_block_match:
            query_text = sql_block_match.group(1).strip()
            print(f"[DEBUG] Extracted from code block: {repr(query_text)}")
        else:
            # Step 3: If no block, try to remove just the markers
            print("[DEBUG] No code block found, removing just the markers")
            query_text = re.sub(r'```sql', '', query_text, flags=re.IGNORECASE)
            query_text = re.sub(r'```', '', query_text)
            print(f"[DEBUG] After removing markers: {repr(query_text)}")
        
        # Step 4: Remove any trailing semicolon for SQLAlchemy
        print("[DEBUG] Step 4: Removing trailing semicolon if present")
        query_text = re.sub(r';\s*$', '', query_text.strip())
        print(f"[DEBUG] Final cleaned query: {repr(query_text)}")
        return query_text
    
    def _validate_sql_query(self, query: str) -> Tuple[bool, str]:
        """
        Validate SQL query for basic syntax and security.
        Returns (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Empty query"
        
        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False, f"Dangerous operation detected: {keyword}"
        
        # Basic SQL validation - should start with SELECT
        if not query_upper.strip().startswith('SELECT'):
            return False, "Query must start with SELECT"
        
        # Check for balanced parentheses
        if query.count('(') != query.count(')'):
            return False, "Unbalanced parentheses"
        
        return True, "Valid"
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get detailed schema information for the database."""
        try:
            inspector = inspect(self.engine)
            schema_info = {}
            
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                schema_info[table_name] = {
                    "columns": [
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "nullable": col["nullable"],
                            "default": col["default"]
                        }
                        for col in columns
                    ],
                    "primary_keys": inspector.get_pk_constraint(table_name)["constrained_columns"],
                    "foreign_keys": [
                        {
                            "constrained_columns": fk["constrained_columns"],
                            "referred_table": fk["referred_table"],
                            "referred_columns": fk["referred_columns"]
                        }
                        for fk in inspector.get_foreign_keys(table_name)
                    ]
                }
            
            return schema_info
            
        except Exception as e:
            self.logger.error(f"Failed to get schema info: {e}")
            return {}

    def generate_and_execute_query(self, question: str) -> Dict[str, Any]:
        """
        Generate SQL query from natural language and execute it.
        Returns structured response with query, results, and metadata.
        """
        cleaned_query: Optional[str] = None  # Initialize with type hint

        try:
            self.logger.info(f"Processing question: {question}")
            
            # Generate SQL query using the chain
            raw_response = self.db_chain.invoke({"query": question})
            
            # Extract the query from the response
            if isinstance(raw_response, dict):
                query_text = raw_response.get('result', '')
            else:
                query_text = str(raw_response)
            
            self.logger.info(f"Raw LLM response: {query_text[:200]}...")
            self.logger.info(f"FULL RAW RESPONSE: {repr(query_text)}")
            
            # Use a more aggressive approach to clean SQL
            cleaned_query = self._really_clean_sql(query_text)
            
            self.logger.info(f"CLEANED QUERY: {repr(cleaned_query)}")
            
            # Validate the query
            is_valid, validation_message = self._validate_sql_query(cleaned_query)
            if not is_valid:
                return {
                    "success": False,
                    "error": f"Invalid query: {validation_message}",
                    "query": cleaned_query,
                    "results": None
                }
            
            # Execute the query
            with self.engine.connect() as connection:
                result = connection.execute(text(cleaned_query))
                rows = result.fetchall()
                columns = result.keys()
                
                # Convert to list of dictionaries
                results = [dict(zip(columns, row)) for row in rows]
            
            return {
                "success": True,
                "query": cleaned_query,
                "results": results,
                "row_count": len(results),
                "columns": list(columns)
            }
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Query execution failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "query": cleaned_query,
                "results": None
            }

    def _really_clean_sql(self, text: str) -> str:
        """
        Aggressively clean SQL query, guaranteed to remove all markdown.
        """
        self.logger.info(f"Starting aggressive cleaning of: {repr(text[:100])}...")
        
        # First, handle the SQLQuery: prefix
        text = re.sub(r'^SQLQuery:\s*', '', text, flags=re.IGNORECASE)
        
        # Extract SQL from code block if possible
        sql_match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            cleaned = sql_match.group(1).strip()
            self.logger.info(f"Extracted from code block: {repr(cleaned[:100])}...")
        else:
            # Brute force removal of markdown
            cleaned = text
            
            # Remove all variations of markdown code blocks
            for pattern in [
                r'```sql\s*\n?', r'```\s*\n?', r'\n?\s*```\s*$', 
                r'^```sql\s*', r'^```\s*', r'\s*```$', r'```'
            ]:
                cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
            
            self.logger.info(f"After removing markdown: {repr(cleaned[:100])}...")
        
        # Remove trailing semicolon
        cleaned = re.sub(r';\s*$', '', cleaned.strip())
        
        # Make sure query starts with SELECT
        if not re.match(r'^\s*SELECT', cleaned, re.IGNORECASE):
            select_match = re.search(r'(SELECT\s+.*)', cleaned, re.IGNORECASE | re.DOTALL)
            if select_match:
                cleaned = select_match.group(1)
                self.logger.info(f"Extracted query starting with SELECT: {repr(cleaned[:100])}...")
        
        self.logger.info(f"Final cleaned query: {repr(cleaned)}")
        return cleaned.strip()
