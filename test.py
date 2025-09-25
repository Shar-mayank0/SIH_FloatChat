from core.query_engine import QueryEngine


engine = QueryEngine()

query = input("Enter your query: ")
response = engine.send_groq_request(query)
print(response)
print("Response received.")

print("getting sql query from response...")
sql_query = engine.get_sql_from_response(response)
print(f"Extracted SQL query: {sql_query}")

print("Executing SQL query...")
results = engine.execute_sql(sql_query)
print("Query Results:")
for row in results:
    print(row)
    
