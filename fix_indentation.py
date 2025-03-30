#!/usr/bin/env python3

def fix_file():
    with open('C25D.py', 'r') as file:
        content = file.read()
    
    # Split the content into lines for processing
    lines = content.split('\n')
    
    # Fix the specific indentation issues
    
    # Fix at line 1797 (crossing loop)
    i = 1796  # Indexing starts at 0
    if i < len(lines):
        # Check if this is the problematic line
        if "for crossing in self.pedestrian_crossings.values():" in lines[i]:
            # Fix indentation for this line
            indent_level = 8  # Standard method indentation level
            if lines[i].lstrip().startswith("for"):
                lines[i] = ' ' * indent_level + lines[i].lstrip()
                
                # Also fix indentation for the following block
                j = i + 1
                while j < len(lines) and j < i + 20:
                    if lines[j].strip() and not lines[j].startswith(' ' * indent_level):
                        # This line is part of the block and needs indentation
                        if lines[j].strip().startswith('if '):
                            # If statement should be indented within the for loop
                            lines[j] = ' ' * (indent_level + 4) + lines[j].lstrip()
                            
                            # Next line might need further indentation
                            if j + 1 < len(lines) and 'if not emergency_preemption_active' in lines[j + 1]:
                                lines[j + 1] = ' ' * (indent_level + 8) + lines[j + 1].lstrip()
                                
                                # And further for the inner if statement
                                if j + 2 < len(lines) and 'if (' in lines[j + 2]:
                                    lines[j + 2] = ' ' * (indent_level + 12) + lines[j + 2].lstrip()
                                    
                                    # Fix indentation for the indented block within this if
                                    k = j + 3
                                    while k < len(lines) and k < j + 10:
                                        if lines[k].strip() and not lines[k].startswith(' ' * (indent_level + 12)):
                                            if not any(end_marker in lines[k] for end_marker in ['def ', 'class ']):
                                                lines[k] = ' ' * (indent_level + 16) + lines[k].lstrip()
                                        else:
                                            break
                                        k += 1
                        else:
                            # Other lines
                            lines[j] = ' ' * (indent_level + 4) + lines[j].lstrip()
                    else:
                        # Check if we're done with this block
                        if lines[j].strip() and lines[j].startswith(' ' * 4) and any(
                                end_marker in lines[j] for end_marker in ['def ', 'class ']):
                            break
                    j += 1
    
    # Fix issue at line 1809 (update_pedestrians_and_bicycles method)
    i = 1808  # Line 1809 in 0-indexed
    if i < len(lines) and "def update_pedestrians_and_bicycles" in lines[i]:
        # Fix the method definition indentation
        lines[i] = '    ' + lines[i].lstrip()
        
        # Fix indentation for the method body
        j = i + 1
        while j < len(lines) and j < i + 30:
            if lines[j].strip():
                if lines[j].startswith(' ' * 4) and any(end_marker in lines[j] for end_marker in ['def ', 'class ']):
                    # Found next method, stop fixing
                    break
                else:
                    # Indent this line properly (8 spaces for method body)
                    if not lines[j].startswith(' ' * 8):
                        lines[j] = ' ' * 8 + lines[j].lstrip()
            j += 1
    
    # Fix issue at line ~2378 (Road Type visualization)
    i = 2376  # Line 2377 in 0-indexed
    if i < len(lines) and "if 'road_type' in df.columns:" in lines[i]:
        # Get the next line index
        j = i + 1
        # Check if the next line doesn't have correct indentation
        if j < len(lines) and lines[j].strip() and not lines[j].startswith(' ' * 4):
            # Calculate how many spaces should be used for indentation
            indent_level = 4
            # Indent all lines in this block
            while j < len(lines) and j < i + 20:
                if lines[j].strip() and "else:" in lines[j]:
                    # Found the else statement, stop fixing
                    break
                # Add proper indentation
                lines[j] = ' ' * indent_level + lines[j].lstrip()
                j += 1
    
    # Fix issue at line ~2417 (Congestion Level visualization)
    i = 2395
    if i < len(lines) and "if 'congestion_level' in df.columns:" in lines[i]:
        # Get the next line index
        j = i + 1
        # Check if the next line doesn't have correct indentation
        if j < len(lines) and lines[j].strip() and not lines[j].startswith(' ' * 4):
            # Calculate how many spaces should be used for indentation
            indent_level = 4
            # Indent all lines in this block
            while j < len(lines) and j < i + 20:
                if lines[j].strip() and "else:" in lines[j]:
                    # Found the else statement, stop fixing
                    break
                # Add proper indentation
                lines[j] = ' ' * indent_level + lines[j].lstrip()
                j += 1
    
    # Write the fixed content back to file
    with open('C25D.py', 'w') as file:
        file.write('\n'.join(lines))
    
    print("Fixed indentation issues in C25D.py")

if __name__ == "__main__":
    fix_file()
