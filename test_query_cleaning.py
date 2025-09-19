#!/usr/bin/env python3
"""
Test script to verify the SQL cleaning function works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test the cleaning function without needing full database setup
import re
import logging

def clean_sql_response(response: str) -> str:
    """
    Clean and extract SQL query from the LLM response.
    Removes markdown formatting and extracts clean SQL.
    """
    if not response:
        return ""
    
    print(f"Raw response: {repr(response)}")
    
    # Step 1: Remove markdown code blocks - be more aggressive
    cleaned = response.strip()
    
    # Remove ```sql from beginning (case insensitive)
    cleaned = re.sub(r'^\s*```\s*sql\s*\n?', '', cleaned, flags=re.IGNORECASE)
    
    # Remove ``` from end
    cleaned = re.sub(r'\n?\s*```\s*$', '', cleaned, flags=re.IGNORECASE)
    
    # Remove any remaining ``` markers anywhere in the text
    cleaned = re.sub(r'```\s*sql\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'```', '', cleaned)
    
    print(f"After removing backticks: {repr(cleaned)}")
    
    # Step 2: Remove common LLM response prefixes
    cleaned = re.sub(r'^\s*(?:SQL\s*Query:\s*|Query:\s*|Here.*?query:\s*)', '', cleaned, flags=re.IGNORECASE)
    
    # Step 3: Remove backticks
    cleaned = re.sub(r'`+', '', cleaned)
    
    # Step 4: Simple approach - if it looks like SQL, use it directly
    cleaned = cleaned.strip()
    
    # If it starts with SELECT (case insensitive), it's probably valid SQL
    if re.match(r'^\s*SELECT\s+', cleaned, re.IGNORECASE):
        # Remove trailing semicolon for SQLAlchemy
        cleaned = re.sub(r';\s*$', '', cleaned)
        print(f"Found SQL starting with SELECT: {repr(cleaned[:100])}...")
        return cleaned.strip()
    
    # Fallback: try to extract SQL lines manually
    lines = cleaned.split('\n')
    sql_lines = []
    in_sql_block = False
    
    for line in lines:
        stripped_line = line.strip()
        
        # Skip empty lines if we haven't started SQL yet
        if not stripped_line and not in_sql_block:
            continue
        
        # Check if this line starts SQL
        if re.match(r'^\s*(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s+', stripped_line, re.IGNORECASE):
            in_sql_block = True
            sql_lines.append(line)
            print(f"Found SQL start: {stripped_line}")
        elif in_sql_block:
            # Stop if we hit explanatory text
            if re.match(r'^\s*(?:This|The|Here|Note|Explanation|Result|Answer|Query|SQL).*', stripped_line, re.IGNORECASE):
                print(f"Stopping at explanatory text: {stripped_line}")
                break
            # Continue adding SQL lines
            sql_lines.append(line)
    
    # Join the SQL lines
    cleaned_sql = '\n'.join(sql_lines).strip()
    
    # Remove trailing semicolon for SQLAlchemy
    cleaned_sql = re.sub(r';\s*$', '', cleaned_sql)
    
    print(f"Final cleaned SQL: {repr(cleaned_sql)}")
    
    return cleaned_sql.strip()

def test_cleaning():
    """Test the SQL cleaning function with the exact problematic response."""
    test_response = """```sql
SELECT
    ap.profile_id,
    ap.platform_number,
    ap.profile_date,
    m.psal_adjusted
FROM
    measurements m
INNER JOIN
    argo_profiles ap ON m.profile_id = ap.profile_id
WHERE
    m.psal_adjusted > 35
    AND m.psal_adjusted_qc = '1'
ORDER BY
    m.psal_adjusted DESC
LIMIT 100;
```"""
    
    print("Testing SQL cleaning function...")
    print("=" * 50)
    
    cleaned = clean_sql_response(test_response)
    
    print("=" * 50)
    print("Results:")
    
    # Test if cleaned SQL is valid (no ``` markers)
    has_backticks = '```' in cleaned
    print(f"Contains backticks: {has_backticks}")
    
    # Check if it starts with SELECT
    starts_with_select = cleaned.strip().upper().startswith('SELECT')
    print(f"Starts with SELECT: {starts_with_select}")
    
    # Check length
    print(f"Cleaned SQL length: {len(cleaned)}")
    
    if not has_backticks and starts_with_select and len(cleaned) > 0:
        print("✅ Cleaning function working correctly!")
        print("\nCleaned SQL:")
        print(cleaned)
    else:
        print("❌ Cleaning function needs adjustment")
        
    return cleaned

if __name__ == "__main__":
    test_cleaning()