#!/usr/bin/env python3

import re

def clean_file(file_path, output_path=None):
    """
    Remove excess empty lines from a Python file while preserving indentation.
    
    This function:
    1. Removes consecutive empty lines, keeping only one
    2. Maintains indentation of code blocks
    3. Preserves a single blank line between function/class definitions
    4. Maintains a single blank line after imports
    """
    if output_path is None:
        output_path = file_path
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Make a backup
    with open(f"{file_path}.bak", 'w') as f:
        f.write(content)
    
    # Split the content into lines
    lines = content.split('\n')
    
    # Process lines to remove excess empty lines
    cleaned_lines = []
    previous_line_empty = False
    inside_import_block = False
    
    for i, line in enumerate(lines):
        # Check if the current line is empty
        is_empty = not line.strip()
        
        # Check if we're in an import block
        if re.match(r'^\s*import\s+', line) or re.match(r'^\s*from\s+.*import', line):
            inside_import_block = True
        elif line.strip() and not re.match(r'^\s*#', line) and inside_import_block:
            # First non-comment, non-empty line after imports
            inside_import_block = False
            # Make sure we have exactly one blank line after imports
            if cleaned_lines and not previous_line_empty:
                cleaned_lines.append('')
                previous_line_empty = True
        
        # For consecutive empty lines, keep only one
        if is_empty:
            if not previous_line_empty:
                cleaned_lines.append(line)
                previous_line_empty = True
        else:
            cleaned_lines.append(line)
            previous_line_empty = False
    
    # Join the cleaned lines back into a single string
    cleaned_content = '\n'.join(cleaned_lines)
    
    # Write the cleaned content back to the file
    with open(output_path, 'w') as f:
        f.write(cleaned_content)
    
    # Count the reduction in lines
    lines_removed = len(lines) - len(cleaned_lines)
    print(f"Removed {lines_removed} excess empty lines from {file_path}")
    return lines_removed

if __name__ == "__main__":
    import sys
    
    file_path = "/Users/sahand/Desktop/C25D/C25D.py"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    lines_removed = clean_file(file_path)
    print(f"File cleaned successfully. {lines_removed} excess lines removed.") 