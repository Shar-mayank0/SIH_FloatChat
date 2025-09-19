#!/usr/bin/env python3
"""
Integration test for the fixed QueryEngine SQL cleaning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the actual QueryEngine class to test the cleaning method
from core.query_engine import QueryEngine

def test_query_engine_cleaning():
    """Test the QueryEngine._clean_sql_response method directly."""
    try:
        # Create a QueryEngine instance
        # Note: This will fail if GROQ_API_KEY is not set, but we only need the cleaning method
        query_engine = QueryEngine()
        
        # Test the problematic response
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
        
        print("Testing QueryEngine._clean_sql_response method...")
        print("=" * 60)
        
        cleaned = query_engine._clean_sql_response(test_response)
        
        print("=" * 60)
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
            print("✅ QueryEngine cleaning function working correctly!")
            print("\nCleaned SQL:")
            print(cleaned)
            return True
        else:
            print("❌ QueryEngine cleaning function needs adjustment")
            return False
            
    except Exception as e:
        print(f"Error testing QueryEngine: {e}")
        print("This might be due to missing environment variables, but that's okay for this test.")
        return False

def test_built_in_cleaning():
    """Test the built-in test_cleaning method if it exists."""
    try:
        query_engine = QueryEngine()
        if hasattr(query_engine, 'test_cleaning'):
            print("\nTesting built-in test_cleaning method...")
            result = query_engine.test_cleaning()
            print(f"Built-in test result: {repr(result)}")
            return True
        else:
            print("No built-in test_cleaning method found.")
            return False
    except Exception as e:
        print(f"Error testing built-in method: {e}")
        return False

if __name__ == "__main__":
    print("Testing QueryEngine SQL Cleaning Fix")
    print("=" * 60)
    
    # Test 1: Direct method test
    success1 = test_query_engine_cleaning()
    
    # Test 2: Built-in test method
    success2 = test_built_in_cleaning()
    
    print("\n" + "=" * 60)
    if success1:
        print("✅ MAIN TEST PASSED: SQL cleaning is working correctly!")
        print("The syntax error issue should now be resolved.")
    else:
        print("❌ MAIN TEST FAILED: SQL cleaning needs more work.")
        
    print("\nSummary of changes made:")
    print("1. Improved regex patterns to remove ```sql and ``` markers")
    print("2. Added fallback logic for different response formats")
    print("3. Better handling of multiline SQL queries")
    print("4. Enhanced logging for debugging")
    print("5. Removed trailing semicolons (SQLAlchemy requirement)")