#!/usr/bin/env python3

def fix_file():
    # Read the file content as a single string to handle line continuations
    with open('C25D.py', 'r') as file:
        content = file.read()
    
    # Fix specific problematic blocks by pattern replacement
    
    # Pattern 1: Fix all possible "if condition:" blocks without proper indentation
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        fixed_lines.append(lines[i])
        
        # Look for if statements
        if lines[i].strip().startswith('if ') and lines[i].strip().endswith(':'):
            indent_level = len(lines[i]) - len(lines[i].lstrip())
            
            # Check next line for proper indentation
            if i + 1 < len(lines) and lines[i+1].strip():
                next_indent = len(lines[i+1]) - len(lines[i+1].lstrip())
                
                # If next line has same or less indentation, it needs to be fixed
                if next_indent <= indent_level:
                    # Find the end of this if block
                    j = i + 1
                    while j < len(lines):
                        if not lines[j].strip():
                            j += 1
                            continue
                        
                        curr_indent = len(lines[j]) - len(lines[j].lstrip())
                        
                        # Found else or line with same/less indentation that's not part of block
                        if (curr_indent <= indent_level and (
                            lines[j].strip().startswith('else:') or 
                            lines[j].strip().startswith('elif ') or
                            not any(kw in lines[j].strip() for kw in ['and', 'or', '+', '-', '*', '/', ','])
                        )):
                            break
                            
                        j += 1
                    
                    # Fix indentation for all lines in the block
                    for k in range(i+1, j):
                        if lines[k].strip():
                            fixed_lines[len(fixed_lines)-1] = lines[i]  # Keep the if line as is
                            fixed_lines.append(' ' * (indent_level + 4) + lines[k].lstrip())
                    
                    i = j - 1  # Continue after this block
        
        i += 1
    
    # Special fixes for known problematic sections
    final_content = '\n'.join(fixed_lines)
    
    # Fix road_type visualization
    if "# 6. Performance by Road Type" in final_content:
        road_type_fix = """    # 6. Performance by Road Type
    fig5a = plt.figure(figsize=(10, 6))
    if 'road_type' in df.columns:
        road_data = df.groupby(['control_type', 'road_type'])['wait_time'].mean().reset_index()
        sns.barplot(x='road_type', y='wait_time', hue='control_type',
                    data=road_data, palette=[COLOR_PALETTE['traditional'], COLOR_PALETTE['ai']], dodge=True)
        plt.title('Average Wait Time by Road Type')
        plt.xlabel('Road Type')
    else:
        # Alternative visualization if road_type is missing
        intersection_data = df.groupby(['control_type', 'intersection_id'])['wait_time'].mean().reset_index()
        sns.barplot(x='intersection_id', y='wait_time', hue='control_type',
                    data=intersection_data, palette=[COLOR_PALETTE['traditional'], COLOR_PALETTE['ai']], dodge=True)
        plt.title('Average Wait Time by Intersection')
        plt.xlabel('Intersection ID')"""
        
        road_pattern = r"# 6\. Performance by Road Type.*?plt\.xlabel\('Intersection ID'\)"
        import re
        final_content = re.sub(road_pattern, road_type_fix, final_content, flags=re.DOTALL)
    
    # Fix congestion level visualization
    if "# 7. Performance by Congestion Level" in final_content:
        congestion_fix = """    # 7. Performance by Congestion Level
    fig5b = plt.figure(figsize=(10, 6))
    if 'congestion_level' in df.columns:
        congestion_data = df.groupby(['control_type', 'congestion_level'])['wait_time'].mean().reset_index()
        congestion_order = {'low': 0, 'medium': 1, 'high': 2}
        congestion_data['order'] = congestion_data['congestion_level'].map(congestion_order)
        congestion_data = congestion_data.sort_values('order')
        
        sns.barplot(x='congestion_level', y='wait_time', hue='control_type',
                    data=congestion_data, palette=[COLOR_PALETTE['traditional'], COLOR_PALETTE['ai']], dodge=True)
        plt.title('Average Wait Time by Congestion Level')
        plt.xlabel('Congestion Level')
    else:
        # Alternative visualization if congestion_level is missing
        day_data = df.groupby(['control_type', 'day'])['wait_time'].mean().reset_index()
        sns.barplot(x='day', y='wait_time', hue='control_type',
                    data=day_data, palette=[COLOR_PALETTE['traditional'], COLOR_PALETTE['ai']], dodge=True)
        plt.title('Average Wait Time by Day')
        plt.xlabel('Day')"""
        
        congestion_pattern = r"# 7\. Performance by Congestion Level.*?plt\.xlabel\('Day'\)"
        import re
        final_content = re.sub(congestion_pattern, congestion_fix, final_content, flags=re.DOTALL)
    
    # Add more specific fixes for any remaining problematic sections as needed
    
    # Fix for line 5157 (Impact of Road Type and Congestion)
    impact_roads_fix = """    # 19. Impact of Road Type and Congestion
    fig10b = plt.figure(figsize=(10, 6))
    if 'road_type' in df.columns and 'congestion_level' in df.columns:
        road_congestion = df.groupby(['road_type', 'congestion_level', 'control_type'])['wait_time'].mean().reset_index()
        # Use only our two main colors, creating shades for congestion levels
        congestion_palette = sns.light_palette(COLOR_PALETTE['traditional'], n_colors=5)[1:4]
        
        sns.barplot(data=road_congestion, x='road_type', y='wait_time', hue='congestion_level',
                    palette=congestion_palette)
        plt.title('Impact of Road Type and Congestion')
        plt.xlabel('Road Type')
        plt.ylabel('Average Wait Time (seconds)')
    else:
        # Alternative visualization if road_type or congestion_level is missing
        hour_control = df.groupby(['hour', 'control_type'])['wait_time'].mean().reset_index()
        sns.lineplot(data=hour_control, x='hour', y='wait_time', hue='control_type',
                    markers=True, palette=[COLOR_PALETTE['traditional'], COLOR_PALETTE['ai']])
        plt.title('Wait Times by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Wait Time (seconds)')
        plt.xticks(range(0, 24, 3))
    plt.tight_layout()
    figures.append(fig10b)
    plt.close(fig10b)"""
    
    impact_pattern = r"# 19\. Impact of Road Type and Congestion.*?plt\.close\(fig10b\)"
    import re
    final_content = re.sub(impact_pattern, impact_roads_fix, final_content, flags=re.DOTALL)
    
    # Fix for AI system improvement section
    ai_improvement_fix = """    # 20. AI System Improvement by Congestion Level
    fig10c = plt.figure(figsize=(10, 6))
    if 'congestion_level' in df.columns:
        ai_improvement = df.pivot_table(
            values='wait_time',
            index='congestion_level',
            columns='control_type',
            aggfunc='mean'
        ).reset_index()
        ai_improvement['improvement'] = ((ai_improvement['traditional'] - ai_improvement['ai']) / 
                                        ai_improvement['traditional'] * 100)
        
        # Sort by congestion level
        congestion_order = {'low': 0, 'medium': 1, 'high': 2}
        ai_improvement['order'] = ai_improvement['congestion_level'].map(congestion_order)
        ai_improvement = ai_improvement.sort_values('order')
        
        sns.barplot(data=ai_improvement, x='congestion_level', y='improvement',
                    color=COLOR_PALETTE['ai'])
        plt.title('AI System Improvement by Congestion Level')
        plt.xlabel('Congestion Level')
        plt.ylabel('Improvement Percentage (%)')
    else:
        # Alternative visualization if congestion_level is missing
        ai_improvement_by_hour = df.pivot_table(
            values='wait_time',
            index='hour',
            columns='control_type',
            aggfunc='mean'
        ).reset_index()
        ai_improvement_by_hour['improvement'] = ((ai_improvement_by_hour['traditional'] - ai_improvement_by_hour['ai']) / 
                                                ai_improvement_by_hour['traditional'] * 100)
        
        sns.barplot(data=ai_improvement_by_hour, x='hour', y='improvement',
                    color=COLOR_PALETTE['ai'])
        plt.title('AI System Improvement by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Improvement Percentage (%)')
        plt.xticks(range(0, 24, 3))
    plt.tight_layout()
    figures.append(fig10c)
    plt.close(fig10c)"""
    
    ai_pattern = r"# 20\. AI System Improvement by Congestion Level.*?plt\.close\(fig10c\)"
    final_content = re.sub(ai_pattern, ai_improvement_fix, final_content, flags=re.DOTALL)
    
    # Fix for load_model method
    load_model_fix = """    def load_model(self):
        \"\"\"Load existing model and training history\"\"\"
        try:
            if os.path.exists(self.model_path):
                self.load_state_dict(torch.load(self.model_path))
                if os.path.exists(self.history_path):
                    with open(self.history_path, 'r') as f:
                        self.training_history = json.load(f)
                else:
                    # Initialize with empty lists
                    self.training_history['loss'] = []
                    self.training_history['val_loss'] = []
        except Exception as e:
            # Initialize with empty lists
            self.training_history['loss'] = []
            self.training_history['val_loss'] = []"""
    
    load_model_pattern = r"def load_model\(self\):.*?self\.training_history\['val_loss'\] = \[\]"
    final_content = re.sub(load_model_pattern, load_model_fix, final_content, flags=re.DOTALL)
    
    # Write the fixed content back to file
    with open('C25D.py', 'w') as file:
        file.write(final_content)
    
    print("Fixed all indentation issues in C25D.py")

if __name__ == "__main__":
    fix_file() 