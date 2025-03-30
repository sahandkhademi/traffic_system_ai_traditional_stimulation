#!/bin/bash

# Create a temporary file with the fixed load_model method
cat > fixed_load_model.txt << 'EOT'
    def load_model(self):
        """Load existing model and training history"""
        try:
            if os.path.exists(self.model_path):
                self.load_state_dict(torch.load(self.model_path))
                if os.path.exists(self.history_path):
                    with open(self.history_path, "r") as f:
                        self.training_history = json.load(f)
                else:
                    # Initialize with empty lists
                    self.training_history['loss'] = []
                    self.training_history['val_loss'] = []
        except Exception as e:
            # Initialize with empty lists
            self.training_history['loss'] = []
            self.training_history['val_loss'] = []
EOT

# Find the line where the load_model method starts
start_line=$(grep -n "def load_model" C25D.py | cut -d: -f1)

# Find the line where the next method starts (prepare_inputs)
end_line=$(grep -n "def prepare_inputs" C25D.py | cut -d: -f1)

# Calculate how many lines to remove
lines_to_remove=$((end_line - start_line))

# Use sed to replace the load_model method with the fixed version
sed -i.bak "${start_line},${end_line}s/^.*$//" C25D.py

# Use sed to insert the fixed load_model method at the start_line
sed -i.bak "${start_line}r fixed_load_model.txt" C25D.py

# Clean up
rm fixed_load_model.txt

echo "Fixed load_model method in C25D.py" 