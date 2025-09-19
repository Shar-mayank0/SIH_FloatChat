# main.py
from core.query_engine import QueryEngine
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import os

class ArgoConsole:
    def __init__(self):
    def __init__(self):
        print("[ArgoConsole.__init__] Initializing QueryEngine...")
        self.query_engine = QueryEngine()
        print("[ArgoConsole.__init__] QueryEngine initialized.")
        self.query_history = []
        print("[ArgoConsole.__init__] Query history initialized.")
        self.session_start = datetime.now()
        print(f"[ArgoConsole.__init__] Session started at {self.session_start}")
        
    def clear_screen(self):
        """Clear the console screen."""
        print("[ArgoConsole.clear_screen] Clearing screen...")
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_separator(self, char='-', length=80):
        """Print a separator line."""
        print(f"[ArgoConsole.print_separator] Printing separator with char='{char}' and length={length}")
        print(char * length)
        
    def print_header(self, title):
        """Print a header with a title."""
        print(f"[ArgoConsole.print_header] Printing header for title: {title}")
        self.print_separator()
        print(f"| {title} |".center(80))
        self.print_separator()
        
    def show_welcome(self):
        """Display welcome screen with system information."""
        print("[ArgoConsole.show_welcome] Showing welcome screen...")
        self.print_header("🌊 ARGO Oceanographic Data Chatbot")
        
        welcome_text = """
Welcome to the interactive console for querying Argo oceanographic data!

Available Commands:

Example Queries:
        """
        
        print(welcome_text)
        print()
        
        # Show system status
        self.show_system_status()
    
    def show_system_status(self):
        """Display system status information."""
        print("[ArgoConsole.show_system_status] Showing system status...")
        self.print_header("🔧 System Status")
        
        try:
            # Test database connection by getting schema info
            schema = self.query_engine.get_schema_info()
            if schema:
                db_status = "✅ Connected"
                table_count = len(schema)
            else:
                db_status = "⚠️ Connected (No schema info)"
                table_count = 0
        except Exception as e:
            db_status = f"❌ Error: {str(e)[:50]}..."
            table_count = 0
        
        print(f"Database Connection: {db_status}")
        if table_count > 0:
            print(f"Available Tables: ✅ {table_count} tables found")
        print(f"LLM Model: ✅ Llama-3.3-70B (Groq)")
        print(f"Session Started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def show_schema(self):
        """Display database schema in a formatted way."""
        print("[ArgoConsole.show_schema] Showing database schema...")
        self.print_header("📊 Database Schema")
        
        try:
            schema_info = self.query_engine.get_schema_info()
            
            if not schema_info:
                print("No schema information available")
                return
            
            for table_name, table_info in schema_info.items():
                print(f"\nTable: {table_name}")
                print("=" * len(f"Table: {table_name}"))
                
                # Add columns
                print("\nColumns:")
                for col in table_info.get("columns", []):
                    nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
                    default = f" DEFAULT {col['default']}" if col.get("default") else ""
                    print(f"- {col['name']} ({col['type']}) {nullable}{default}")
                
                # Add primary keys
                if table_info.get("primary_keys"):
                    print(f"\nPrimary Keys: {', '.join(table_info['primary_keys'])}")
                
                # Add foreign keys
                if table_info.get("foreign_keys"):
                    print("\nForeign Keys:")
                    for fk in table_info['foreign_keys']:
                        print(f"- {', '.join(fk['constrained_columns'])} → {fk['referred_table']}.{', '.join(fk['referred_columns'])}")
                
                self.print_separator()
            
        except Exception as e:
            print(f"Error retrieving schema: {str(e)}")
    
    def show_sample_data(self, table_name: Optional[str] = None):
        """Display sample data from specified table."""
            print(f"[ArgoConsole.show_sample_data] Showing sample data for table: {table_name}")
            if not table_name:
                print("Available tables: argo_profiles, measurements")
                table_name = input("Enter table name [argo_profiles]: ").strip() or "argo_profiles"
        
        # Create a simple query to get sample data
        sample_query = f"Show me 5 sample records from the {table_name} table"
        
        print(f"Fetching sample data from {table_name}...")
        result = self.query_engine.generate_and_execute_query(sample_query)
        
        if result["success"]:
            self._display_query_results(result, title=f"📋 Sample Data from {table_name}")
        else:
            print(f"Error: {result['error']}")
    
    def show_history(self):
        """Display query history."""
        print("[ArgoConsole.show_history] Showing query history...")
        self.print_header("📜 Query History (Last 10)")
        
        if not self.query_history:
            print("No queries in history")
            return
        
        print(f"{'#':<4} {'Query':<60} {'Status':<10} {'Rows':<8} {'Time':<12}")
        self.print_separator('-', 94)
        
        for i, entry in enumerate(self.query_history[-10:], 1):  # Show last 10 queries
            status = "Success" if entry["success"] else "Error"
            row_count = str(entry.get("row_count", "N/A")) if entry["success"] else "-"
            
            query_text = entry["query"]
            if len(query_text) > 57:
                query_text = query_text[:57] + "..."
                
            print(f"{i:<4} {query_text:<60} {status:<10} {row_count:<8} {entry['timestamp'].strftime('%H:%M:%S')}")
    
    def show_stats(self):
        """Display session statistics."""
        print("[ArgoConsole.show_stats] Showing session statistics...")
        self.print_header("📈 Session Statistics")
        
        total_queries = len(self.query_history)
        successful_queries = sum(1 for q in self.query_history if q["success"])
        failed_queries = total_queries - successful_queries
        
        # Calculate total rows returned
        total_rows = sum(q.get("row_count", 0) for q in self.query_history if q["success"])
        
        uptime = datetime.now() - self.session_start
        uptime_str = f"{int(uptime.total_seconds() // 60)}m {int(uptime.total_seconds() % 60)}s"
        
        print(f"{'Metric':<20} {'Value'}")
        self.print_separator('-', 30)
        print(f"{'Session Uptime':<20} {uptime_str}")
        print(f"{'Total Queries':<20} {total_queries}")
        print(f"{'Successful Queries':<20} {successful_queries}")
        print(f"{'Failed Queries':<20} {failed_queries}")
        print(f"{'Total Rows Retrieved':<20} {total_rows:,}")
        
        if total_queries > 0:
            success_rate = (successful_queries / total_queries) * 100
            print(f"{'Success Rate':<20} {success_rate:.1f}%")
    
    def _display_query_results(self, result: Dict[str, Any], title: str = "🎯 Query Results"):
        """Display query results in a formatted table."""
            print(f"[ArgoConsole._display_query_results] Displaying query results for title: {title}")
            print(f"[ArgoConsole._display_query_results] Result: {result}")
            if not result["success"]:
                print("[ArgoConsole._display_query_results] Result not successful, returning.")
                return
        
        results = result["results"]
        row_count = result["row_count"]
        
        if not results:
            print("Query executed successfully but returned no results.")
            return
        
        # Show the SQL query that was executed
        if result.get("query"):
            self.print_header("🔍 Generated SQL Query")
            print(result["query"])
            print()
        
        # Display results
        self.print_header(title)
        
        # Get columns
        columns = result.get("columns", [])
        if not columns and results:
            columns = list(results[0].keys())
        
        # Determine column widths (limit to reasonable size)
        col_widths = {}
        for col in columns:
            col_widths[col] = min(max(len(str(col)), 5), 20)
            
        for row in results[:50]:  # Only check first 50 rows for width calculation
            for col in columns:
                if isinstance(row, dict):
                    val = str(row.get(col, ""))[:20]  # Limit column content to 20 chars
                else:
                    val = str(row[columns.index(col)])[:20]
                col_widths[col] = min(max(col_widths[col], len(val)), 20)
        
        # Print header
        header = " | ".join(str(col).ljust(col_widths[col]) for col in columns)
        print(header)
        print("-" * len(header))
        
        # Print rows
        display_limit = 50
        for i, row in enumerate(results[:display_limit]):
            if isinstance(row, dict):
                values = [str(row.get(col, ""))[:col_widths[col]].ljust(col_widths[col]) for col in columns]
            else:
                values = [str(val)[:col_widths[col]].ljust(col_widths[col]) for col, val in zip(columns, row)]
            print(" | ".join(values))
        
        # Show truncation message if needed
        if row_count > display_limit:
            print(f"\nShowing first {display_limit} of {row_count:,} total rows")
        else:
            print(f"\nTotal: {row_count:,} rows")
    
    def process_query(self, user_input: str):
        """Process a natural language query."""
            print(f"[ArgoConsole.process_query] Received user_input: {user_input}")
            print("🤖 Processing your query...", end="", flush=True)
        
        # Execute query using the query engine
            print(f"[ArgoConsole.process_query] Calling generate_and_execute_query...")
            result = self.query_engine.generate_and_execute_query(user_input)
            print(f"[ArgoConsole.process_query] Query result: {result}")
        
        # Add debugging output
        if not result["success"] and "syntax error" in str(result.get("error", "")):
            query_text = str(result.get("query", ""))
            print("\nDebug - Raw Query:")
            print(f"{repr(query_text)}")
        
        # Record in history
            print(f"[ArgoConsole.process_query] Recording query history entry...")
        history_entry = {
            "query": user_input,
            "success": result["success"],
            "timestamp": datetime.now(),
            "row_count": result.get("row_count", 0) if result["success"] else 0,
            "error": result.get("error") if not result["success"] else None
        }
        self.query_history.append(history_entry)
        
            print("[ArgoConsole.process_query] Query history updated.")
        print(" ✅ Done!")
        
        # Display results
        if result["success"]:
                print("[ArgoConsole.process_query] Displaying query results...")
            self._display_query_results(result)
        else:
                print("[ArgoConsole.process_query] Displaying error...")
            self._display_error(result, user_input)
    
    def _display_error(self, result: Dict[str, Any], user_input: str):
        """Display error information with helpful suggestions."""
        print(f"[ArgoConsole._display_error] Displaying error for user_input: {user_input}")
        print(f"[ArgoConsole._display_error] Result: {result}")
        error_message = result["error"]
        query = result.get("query", "")
        
        # Determine error type and provide appropriate messaging
        if "syntax error" in error_message.lower():
            error_type = "SQL Syntax Error"
            suggestions = self._get_syntax_error_suggestions(error_message)
        elif "does not exist" in error_message.lower():
            error_type = "Database Schema Error"
            suggestions = self._get_schema_error_suggestions(error_message)
        elif "connection" in error_message.lower():
            error_type = "Database Connection Error"
            suggestions = "• Check database connection\n• Try restarting the application"
        else:
            error_type = "Query Processing Error"
            suggestions = self._get_general_error_suggestions()
        
        self.print_header("❌ Query Failed")
        print(f"{error_type}:")
        print(error_message)
        print()
        
        if query:
            print("Generated SQL Query:")
            print(query)
            print()
        
        if suggestions:
            print("💡 Suggestions:")
            print(suggestions)
    
    def _get_syntax_error_suggestions(self, error_message: str) -> str:
        """Get suggestions for SQL syntax errors."""
        print(f"[ArgoConsole._get_syntax_error_suggestions] Error message: {error_message}")
        if "```" in error_message:
            return ("• The query contained markdown formatting\n"
                   "• This is a system issue - please try rephrasing your question\n"
                   "• Be more specific about what data you want to retrieve")
        return ("• Try rephrasing your question more clearly\n"
               "• Be specific about which data fields you want\n"
               "• Use simpler language to describe your query")
    
    def _get_schema_error_suggestions(self, error_message: str) -> str:
        """Get suggestions for schema-related errors."""
        print(f"[ArgoConsole._get_schema_error_suggestions] Error message: {error_message}")
        return ("• Use '/schema' command to see available tables and columns\n"
               "• Available tables: argo_profiles, measurements\n"
               "• Check column names and table references")
    
    def _get_general_error_suggestions(self) -> str:
        """Get general error suggestions."""
        print("[ArgoConsole._get_general_error_suggestions] Getting general error suggestions...")
        return ("• Try rephrasing your question\n"
               "• Be more specific about what data you want\n"
               "• Use '/help' to see example queries\n"
               "• Use '/schema' to explore available data")
    
    def handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True if should continue, False if should exit."""
        print(f"[ArgoConsole.handle_command] Handling command: {command}")
        command = command.strip().lower()
        
        if command == "/exit":
            print("Are you sure you want to exit? (y/n)", end=" ")
            if input().lower() in ['y', 'yes']:
                print("Thank you for using Argo Data Explorer! 🌊")
                return False
        
        elif command == "/help":
            self.show_welcome()
        
        elif command == "/schema":
            self.show_schema()
        
        elif command.startswith("/sample"):
            parts = command.split()
            table_name = parts[1] if len(parts) > 1 else None
            self.show_sample_data(table_name)
        
        elif command == "/history":
            self.show_history()
        
        elif command == "/clear":
            self.clear_screen()
            self.show_welcome()
        
        elif command == "/stats":
            self.show_stats()
        
        else:
            print(f"Unknown command: {command}")
            print("Type /help for available commands")
        
        return True
    
    def run(self):
        """Main console loop."""
        print("[ArgoConsole.run] Starting main console loop...")
        self.clear_screen()
        self.show_welcome()
        
        try:
            while True:
                print()  # Add spacing
                
                # Get user input
                user_input = input("🤖 Argo Bot> ").strip()
                
                if not user_input:
                    continue
                
                # Check if it's a command
                if user_input.startswith('/'):
                    if not self.handle_command(user_input):
                        break
                else:
                    # Process as natural language query
                    print(f"[ArgoConsole.process_query] Received user_input: {user_input}")
                    print("🤖 Processing your query...", end="", flush=True)
                    print(f"[ArgoConsole.process_query] Calling generate_and_execute_query...")
                    result = self.query_engine.generate_and_execute_query(user_input)
                    print(f"[ArgoConsole.process_query] Query result: {result}")
                    # Add debugging output
                    if not result["success"] and "syntax error" in str(result.get("error", "")):
                        query_text = str(result.get("query", ""))
                        print("\nDebug - Raw Query:")
                        print(f"{repr(query_text)}")
                    # Record in history
                    print(f"[ArgoConsole.process_query] Recording query history entry...")
                    history_entry = {
                        "query": user_input,
                        "success": result["success"],
                        "timestamp": datetime.now(),
                        "row_count": result.get("row_count", 0) if result["success"] else 0,
                        "error": result.get("error") if not result["success"] else None
                    }
                    self.query_history.append(history_entry)
                    print("[ArgoConsole.process_query] Query history updated.")
                    print(" ✅ Done!")
                    # Display results
                    if result["success"]:
                        print("[ArgoConsole.process_query] Displaying query results...")
                        self._display_query_results(result)
                    else:
                        print("[ArgoConsole.process_query] Displaying error...")
                        self._display_error(result, user_input)