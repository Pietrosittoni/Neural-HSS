#!/bin/bash

# Path to your base config and main script
BASE_CONFIG="config/base.yaml"
MAIN_SCRIPT="main.py"
TEMP_CONFIG="config/temp_config.yaml"

# Arrays of values to sweep over
TRAIN_SIZES=(100 250 500 750 100)
SPATIAL_RESOLUTIONS=(64 128 256 512)
MODELS=(mlp)
hidden_channels=(64 128 256)
hidden_layers=(2 4 6)

for train_size in "${TRAIN_SIZES[@]}"; do
  for spatial_resolution in "${SPATIAL_RESOLUTIONS[@]}"; do
    for model in "${MODELS[@]}"; do
      for hidden_channel in "${hidden_channels[@]}"; do
        for hidden_layer in "${hidden_layers[@]}"; do
          # Copy base config to a temp config
          cp "$BASE_CONFIG" "$TEMP_CONFIG"
          # Edit the temp config with new values
          yq -i ".train_size = $train_size | .spatial_resolution = $spatial_resolution | .model = \"$model\" | .hidden_channels = $hidden_channel | .hidden_layers = $hidden_layer" "$TEMP_CONFIG"
          # Run the experiment
          echo "Running with train_size=$train_size, spatial_resolution=$spatial_resolution, model=$model, hidden_channels=$hidden_channel, hidden_layers=$hidden_layer"
          python "$MAIN_SCRIPT" --config "$TEMP_CONFIG"
        done
      done
    done
  done
done

# Clean up
rm "$TEMP_CONFIG"