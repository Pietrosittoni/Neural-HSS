#!/bin/bash

CONFIG_FILE="config/pois3d.yaml"
MAIN_SCRIPT="main3d.py"
TEMP_CONFIG="config/temp_pois3d.yaml"

# 3D models to run
#MODELS=("fno_3d" "resnet_3d" "deeponet_3d" "hss_mlp_3d")
MODELS=("hss_mlp_3d" "fno_3d" "resnet_3d" "deeponet_3d")


# Function to modify YAML to change model
modify_config() {
    local config_file="$1"
    local model="$2"
    
    # Copy base config
    cp "$CONFIG_FILE" "$config_file"
    
    # Modify model parameter
    sed -i "s/^model:.*/model: \"$model\"/" "$config_file"
}

# Check if base config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Base config file not found: $CONFIG_FILE"
    exit 1
fi

# Create config directory if it doesn't exist
mkdir -p "$(dirname "$TEMP_CONFIG")"

echo "Starting 3D model runs..."

for model in "${MODELS[@]}"; do
    echo "=== Running $model ==="
    
    # Update config with current model
    modify_config "$TEMP_CONFIG" "$model"
    
    # Run the model
    echo "Running $model with configuration from $TEMP_CONFIG"
    python "$MAIN_SCRIPT" --config "$TEMP_CONFIG"
    
    echo "Completed $model"
    echo ""
done

# Clean up
if [ -f "$TEMP_CONFIG" ]; then
    rm "$TEMP_CONFIG"
    echo "Cleaned up temporary config file"
fi

echo "All model runs complete!"
