# main.py
from core.query_engine import QueryEngine
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import os
import sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.layout import Layout
    from rich.live import Live
    from rich.align import Align
    from rich.columns import Columns
    from rich.rule import Rule
    from rich.status import Status
    from rich import box
    from rich.markdown import Markdown
    from rich.tree import Tree
except ImportError:
    print("Rich library not found. Installing...")
    os.system("pip install rich")
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.layout import Layout
    from rich.live import Live
    from rich.align import Align
    from rich.columns import Columns
    from rich.rule import Rule
    from rich.status import Status
    from rich import box
    from rich.markdown import Markdown
    from rich.tree import Tree


class ArgoRichConsole:
    def __init__(self):
        self.console = Console()
        self.query_engine = QueryEngine()
        self.query_history = []
        self.session_start = datetime.now()
        
    def clear_screen(self):
        """Clear the console screen."""
        self.console.clear()
        
    def show_welcome(self):
        """Display welcome screen with system information."""
        # Create the main welcome panel
        welcome_text = """
🌊 **ARGO Oceanographic Data Chatbot**

Welcome to the interactive console for querying Argo oceanographic data!

**Available Commands:**
• Natural language queries - Ask questions about oceanographic data
• `/help` - Show this help message  
• `/schema` - Display database schema
• `/sample [table]` - Show sample data from a table
• `/history` - Show query history
• `/clear` - Clear the console
• `/stats` - Show session statistics  
• `/exit` - Exit the application

**Example Queries:**
• "Show me the latest 10 profiles from the Pacific Ocean"
• "What's the average temperature at 1000 meter depth?"
• "Find profiles with high salinity measurements"
• "Show temperature trends over the last year"
• "Get profiles with salinity greater than 35"
        """
        
        welcome_panel = Panel(
            Markdown(welcome_text),
            title="🌊 Argo Data Explorer",
            title_align="center",
            border_style="cyan",
            padding=(1, 2)
        )
        
        self.console.print(welcome_panel)
        self.console.print()
        
        # Show system status
        self.show_system_status()
    
    def show_system_status(self):
        """Display system status information in a colorful format."""
        try:
            # Test database connection by trying to get table info
            with self.console.status("[bold green]Testing database connection..."):
                time.sleep(0.5)  # Brief pause for effect
                table_info = self.query_engine.db.get_table_info()
                if table_info:
                    db_status = "[green]✅ Connected[/green]"
                    # Count tables by counting lines that start with table names
                    table_count = table_info.count("Table '") if "Table '" in table_info else 2
                else:
                    db_status = "[yellow]⚠️ Connected (No schema info)[/yellow]"
                    table_count = 0
        except Exception as e:
            db_status = f"[red]❌ Error: {str(e)[:50]}...[/red]"
            table_count = 0
        
        # Create status table
        status_table = Table(show_header=False, box=box.SIMPLE)
        status_table.add_column("Metric", style="bold blue", width=20)
        status_table.add_column("Status", width=40)
        
        status_table.add_row("Database Connection:", db_status)
        if table_count > 0:
            status_table.add_row("Available Tables:", f"[green]✅ {table_count} tables found[/green]")
        status_table.add_row("LLM Model:", "[green]✅ Llama-3.3-70B (Groq)[/green]")
        status_table.add_row("Session Started:", f"[cyan]{self.session_start.strftime('%Y-%m-%d %H:%M:%S')}[/cyan]")
        
        status_panel = Panel(
            status_table,
            title="🔧 System Status",
            title_align="center",
            border_style="green",
            padding=(1, 1)
        )
        
        self.console.print(status_panel)
    
    def show_schema(self):
        """Display database schema in a beautiful tree format."""
        self.console.print(Rule("[bold blue]📊 Database Schema[/bold blue]"))
        
        try:
            with self.console.status("[bold green]Loading schema information..."):
                # Get table info from the database directly
                table_info = self.query_engine.db.get_table_info()
            
            if not table_info:
                self.console.print("[red]No schema information available[/red]")
                return
            
            # Create a tree for the schema - simplified version since we have string info
            schema_tree = Tree("🗄️ Database Schema", style="bold blue")
            
            # Parse the table info string to extract table and column information
            lines = table_info.split('\n')
            current_table = None
            table_node = None
            
            for line in lines:
                line = line.strip()
                if line.startswith("Table '") and line.endswith("' has columns:"):
                    # Extract table name
                    current_table = line.split("'")[1]
                    table_node = schema_tree.add(f"[bold green]📋 {current_table}[/bold green]")
                elif line and table_node and not line.startswith("Table"):
                    # This should be a column description
                    table_node.add(f"[cyan]{line}[/cyan]")
            
            # If parsing failed, show raw table info
            if not any(child for child in schema_tree.children):
                schema_tree.add(f"[cyan]{table_info}[/cyan]")
            
            self.console.print(schema_tree)
            
        except Exception as e:
            self.console.print(f"[red]Error retrieving schema: {str(e)}[/red]")
    
    def show_sample_data(self, table_name: Optional[str] = None):
        """Display sample data from specified table."""
        if not table_name:
            self.console.print("[yellow]Available tables: argo_profiles, measurements[/yellow]")
            table_name = Prompt.ask("Enter table name", default="argo_profiles")
        
        # Create a simple query to get sample data
        sample_query = f"Show me 5 sample records from the {table_name} table"
        
        with self.console.status(f"[bold green]Fetching sample data from {table_name}..."):
            try:
                # Use the actual QueryEngine workflow
                response = self.query_engine.send_groq_request(sample_query)
                sql_query = self.query_engine.get_sql_from_response(response)
                results = self.query_engine.execute_sql(sql_query)
                
                # Create result object for display
                result_obj = {
                    "success": True,
                    "results": [dict(row._mapping) for row in results] if results else [],
                    "row_count": len(results) if results else 0,
                    "query": sql_query,
                    "columns": list(results[0]._mapping.keys()) if results else []
                }
                
                self._display_query_results(result_obj, title=f"📋 Sample Data from {table_name}")
                
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")
    
    def show_history(self):
        """Display query history in a beautiful table."""
        self.console.print(Rule("[bold blue]📜 Query History (Last 10)[/bold blue]"))
        
        if not self.query_history:
            self.console.print("[yellow]No queries in history[/yellow]")
            return
        
        history_table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
            title="Query History",
            title_style="bold blue"
        )
        
        history_table.add_column("#", style="dim", width=4)
        history_table.add_column("Query", style="cyan", min_width=40, max_width=60)
        history_table.add_column("Status", justify="center", width=10)
        history_table.add_column("Rows", justify="right", width=8)
        history_table.add_column("Time", style="dim", width=10)
        
        for i, entry in enumerate(self.query_history[-10:], 1):
            status_style = "green" if entry["success"] else "red"
            status_text = "✅ Success" if entry["success"] else "❌ Error"
            row_count = str(entry.get("row_count", "N/A")) if entry["success"] else "-"
            
            query_text = entry["query"]
            if len(query_text) > 57:
                query_text = query_text[:57] + "..."
            
            history_table.add_row(
                str(i),
                query_text,
                f"[{status_style}]{status_text}[/{status_style}]",
                row_count,
                entry['timestamp'].strftime('%H:%M:%S')
            )
        
        self.console.print(history_table)
    
    def show_stats(self):
        """Display session statistics in colorful panels."""
        total_queries = len(self.query_history)
        successful_queries = sum(1 for q in self.query_history if q["success"])
        failed_queries = total_queries - successful_queries
        total_rows = sum(q.get("row_count", 0) for q in self.query_history if q["success"])
        
        uptime = datetime.now() - self.session_start
        uptime_str = f"{int(uptime.total_seconds() // 60)}m {int(uptime.total_seconds() % 60)}s"
        
        # Create statistics panels
        stats_table = Table(show_header=False, box=box.SIMPLE)
        stats_table.add_column("Metric", style="bold blue", width=20)
        stats_table.add_column("Value", style="green", width=20)
        
        stats_table.add_row("Session Uptime", uptime_str)
        stats_table.add_row("Total Queries", str(total_queries))
        stats_table.add_row("Successful Queries", f"[green]{successful_queries}[/green]")
        stats_table.add_row("Failed Queries", f"[red]{failed_queries}[/red]" if failed_queries > 0 else "0")
        stats_table.add_row("Total Rows Retrieved", f"{total_rows:,}")
        
        if total_queries > 0:
            success_rate = (successful_queries / total_queries) * 100
            stats_table.add_row("Success Rate", f"[green]{success_rate:.1f}%[/green]")
        
        stats_panel = Panel(
            stats_table,
            title="📈 Session Statistics",
            title_align="center",
            border_style="blue",
            padding=(1, 1)
        )
        
        self.console.print(stats_panel)
    
    def _display_query_results(self, result: Dict[str, Any], title: str = "🎯 Query Results"):
        """Display query results in a beautiful table format."""
        if not result["success"]:
            return
        
        results = result["results"]
        row_count = result["row_count"]
        
        if not results:
            self.console.print("[yellow]Query executed successfully but returned no results.[/yellow]")
            return
        
        # Show the SQL query that was executed
        if result.get("query"):
            sql_syntax = Syntax(result["query"], "sql", theme="monokai", line_numbers=False)
            sql_panel = Panel(
                sql_syntax,
                title="🔍 Generated SQL Query",
                title_align="center",
                border_style="yellow",
                padding=(1, 1)
            )
            self.console.print(sql_panel)
            self.console.print()
        
        # Create results table
        results_table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
            title=title,
            title_style="bold green"
        )
        
        # Get columns
        columns = result.get("columns", [])
        if not columns and results:
            columns = list(results[0].keys()) if isinstance(results[0], dict) else []
        
        # Add columns to table
        for col in columns:
            results_table.add_column(col, style="cyan", max_width=25)
        
        # Add rows to table
        display_limit = 50
        for i, row in enumerate(results[:display_limit]):
            if isinstance(row, dict):
                values = [str(row.get(col, ""))[:23] + ("..." if len(str(row.get(col, ""))) > 23 else "") for col in columns]
            else:
                values = [str(val)[:23] + ("..." if len(str(val)) > 23 else "") for val in row]
            results_table.add_row(*values)
        
        self.console.print(results_table)
        
        # Show summary
        if row_count > display_limit:
            summary_text = f"[yellow]Showing first {display_limit} of {row_count:,} total rows[/yellow]"
        else:
            summary_text = f"[green]Total: {row_count:,} rows[/green]"
        
        summary_panel = Panel(
            Align.center(summary_text),
            border_style="green",
            padding=(0, 1)
        )
        self.console.print(summary_panel)
    
    def process_query(self, user_input: str):
        """Process a natural language query with beautiful progress indicators."""
        
        # Step 1: Send request to Groq
        with self.console.status("[bold green]🤖 Sending request to AI model..."):
            try:
                response = self.query_engine.send_groq_request(user_input)
                self.console.print("[green]✅ Response received from AI model[/green]")
            except Exception as e:
                self.console.print(f"[red]❌ Error getting AI response: {str(e)}[/red]")
                return
        
        # Step 2: Extract SQL query
        with self.console.status("[bold yellow]🔍 Extracting SQL query from response..."):
            try:
                sql_query = self.query_engine.get_sql_from_response(response)
                if sql_query:
                    self.console.print(f"[green]✅ SQL query extracted[/green]")
                    
                    # Show the extracted SQL with syntax highlighting
                    sql_syntax = Syntax(sql_query, "sql", theme="monokai", line_numbers=False)
                    sql_panel = Panel(
                        sql_syntax,
                        title="🔍 Generated SQL Query",
                        title_align="center",
                        border_style="yellow",
                        padding=(1, 1)
                    )
                    self.console.print(sql_panel)
                else:
                    self.console.print("[red]❌ Could not extract SQL query from response[/red]")
                    return
            except Exception as e:
                self.console.print(f"[red]❌ Error extracting SQL: {str(e)}[/red]")
                return
        
        # Step 3: Execute SQL query
        with self.console.status("[bold blue]⚡ Executing SQL query..."):
            try:
                results = self.query_engine.execute_sql(sql_query)
                row_count = len(results) if results else 0
                self.console.print(f"[green]✅ Query executed successfully ({row_count} rows)[/green]")
                
                # Convert SQLAlchemy Row objects to dictionaries
                if results and hasattr(results[0], '_mapping'):
                    results_data = [dict(row._mapping) for row in results]
                    columns = list(results[0]._mapping.keys()) if results else []
                else:
                    results_data = results
                    columns = []
                
                # Create result object for display
                result_obj = {
                    "success": True,
                    "results": results_data,
                    "row_count": row_count,
                    "query": sql_query,
                    "columns": columns
                }
                
                # Record in history
                history_entry = {
                    "query": user_input,
                    "success": True,
                    "timestamp": datetime.now(),
                    "row_count": row_count,
                    "error": None
                }
                self.query_history.append(history_entry)
                
                # Display results
                self._display_query_results(result_obj)
                
            except Exception as e:
                error_msg = str(e)
                self.console.print(f"[red]❌ Error executing SQL: {error_msg}[/red]")
                
                # Record failed query in history
                history_entry = {
                    "query": user_input,
                    "success": False,
                    "timestamp": datetime.now(),
                    "row_count": 0,
                    "error": error_msg
                }
                self.query_history.append(history_entry)
                
                # Show error details
                self._display_sql_error(error_msg, sql_query, user_input)
    
    def _display_error(self, result: Dict[str, Any], user_input: str):
        """Display error information with helpful suggestions."""
        error_message = result["error"]
        query = result.get("query", "")
        
        error_panel = Panel(
            f"[red]{error_message}[/red]",
            title="❌ Query Failed",
            title_align="center",
            border_style="red",
            padding=(1, 1)
        )
        self.console.print(error_panel)
        
        if query:
            sql_syntax = Syntax(query, "sql", theme="monokai", line_numbers=False)
            query_panel = Panel(
                sql_syntax,
                title="Generated SQL Query",
                title_align="center",
                border_style="yellow",
                padding=(1, 1)
            )
            self.console.print(query_panel)
        
        # Show suggestions
        suggestions = self._get_error_suggestions(error_message)
        if suggestions:
            suggestions_panel = Panel(
                suggestions,
                title="💡 Suggestions",
                title_align="center",
                border_style="blue",
                padding=(1, 1)
            )
            self.console.print(suggestions_panel)
    
    def _display_sql_error(self, error_msg: str, sql_query: str, user_input: str):
        """Display SQL execution error with helpful information."""
        error_panel = Panel(
            f"[red]{error_msg}[/red]",
            title="❌ SQL Execution Error",
            title_align="center",
            border_style="red",
            padding=(1, 1)
        )
        self.console.print(error_panel)
        
        suggestions = """
• Try rephrasing your question more clearly
• Be specific about which data fields you want
• Use '/schema' to see available tables and columns
• Use '/sample [table]' to see sample data
        """
        
        suggestions_panel = Panel(
            suggestions.strip(),
            title="💡 Suggestions",
            title_align="center",
            border_style="blue",
            padding=(1, 1)
        )
        self.console.print(suggestions_panel)
    
    def _get_error_suggestions(self, error_message: str) -> str:
        """Get suggestions based on error type."""
        if "syntax error" in error_message.lower():
            return "• Try rephrasing your question more clearly\n• Be specific about what data you want to retrieve\n• Use simpler language to describe your query"
        elif "does not exist" in error_message.lower():
            return "• Use '/schema' command to see available tables and columns\n• Available tables: argo_profiles, measurements\n• Check column names and table references"
        elif "connection" in error_message.lower():
            return "• Check database connection\n• Try restarting the application"
        else:
            return "• Try rephrasing your question\n• Be more specific about what data you want\n• Use '/help' to see example queries\n• Use '/schema' to explore available data"
    
    def handle_command(self, command: str) -> bool:
        """Handle special commands with beautiful formatting."""
        command = command.strip().lower()
        
        if command == "/exit":
            if Confirm.ask("[yellow]Are you sure you want to exit?[/yellow]"):
                self.console.print("[green]Thank you for using Argo Data Explorer! 🌊[/green]")
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
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("[yellow]Type /help for available commands[/yellow]")
        
        return True
    
    def run(self):
        """Main console loop with beautiful interface."""
        self.clear_screen()
        self.show_welcome()
        
        try:
            while True:
                self.console.print()  # Add spacing
                
                # Get user input with custom prompt
                user_input = Prompt.ask("[bold blue]🤖 Argo Bot[/bold blue]").strip()
                
                if not user_input:
                    continue
                
                # Check if it's a command
                if user_input.startswith('/'):
                    if not self.handle_command(user_input):
                        break
                else:
                    # Process as natural language query
                    self.process_query(user_input)
                    
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted by user[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]Unexpected error: {str(e)}[/red]")
        finally:
            self.console.print("[green]Goodbye! 👋[/green]")


if __name__ == "__main__":
    console_app = ArgoRichConsole()
    console_app.run()