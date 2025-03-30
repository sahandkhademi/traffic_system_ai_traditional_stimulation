#!/bin/bash

# Create a temporary file with the fixed prepare_inputs method
cat > fixed_prepare_inputs.txt << 'EOT'
    def prepare_inputs(self, intersection_state):
        """Convert intersection state to model inputs"""
        # Extract relevant features from intersection state
        features = [
            intersection_state['vehicle_count'],
            intersection_state['wait_time'],
            intersection_state['hour'],
            intersection_state['is_peak_hour'],
            intersection_state['emergency_vehicles'],
            intersection_state['pedestrian_count'],
            intersection_state['bicycle_count'],
            intersection_state['congestion_level'],
            intersection_state['road_type'],
            intersection_state['current_phase_duration'],
            intersection_state['queue_length'],
            intersection_state['avg_speed']
        ]
        return torch.FloatTensor(features).unsqueeze(0)
EOT

# Find the line where the prepare_inputs method starts
start_line=$(grep -n "def prepare_inputs" C25D.py | cut -d: -f1)

# Find the line where the next method starts (update_training_history)
end_line=$(grep -n "def update_training_history" C25D.py | cut -d: -f1)

# Create a sed script to delete all lines between start_line and end_line
sed -i.bak "${start_line},${end_line}d" C25D.py

# Insert the fixed prepare_inputs method at start_line-1
sed -i.bak "$((start_line-1))r fixed_prepare_inputs.txt" C25D.py

# Clean up
rm fixed_prepare_inputs.txt

echo "Fixed prepare_inputs method in C25D.py" 