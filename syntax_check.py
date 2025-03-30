#!/usr/bin/env python3

import ast
import sys

def verify_syntax(filename):
    """Check if the Python file has valid syntax"""
    try:
        with open(filename, 'r') as file:
            source = file.read()
        
        # Parse the file to check for syntax errors
        ast.parse(source)
        print(f"✅ {filename} has valid syntax!")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {filename} at line {e.lineno}, column {e.offset}")
        print(f"   {e.text.strip() if e.text else ''}")
        print(f"   Error message: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Error reading/parsing {filename}: {str(e)}")
        return False

def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'C25D.py'  # Default filename
    
    verify_syntax(filename)

if __name__ == "__main__":
    main()
