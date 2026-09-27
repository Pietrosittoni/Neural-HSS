#!/bin/bash

# CUDA_VISIBLE_DEVICES=3  # Set to the appropriate GPU ID
BASE_CONFIG="config/basethewell.yaml"
MAIN_SCRIPT="main2d.py"
TEMP_CONFIG="config/temp_configthewell.yaml"

TRAIN_SIZES=(960)
SPATIAL_RESOLUTIONS=(128)

# MLP parameters
MLP_HIDDEN_CHANNELS=(16 32 64)
MLP_HIDDEN_LAYERS=(1 2 4)

# FNO parameters
FNO_MODES=(32)
FNO_HIDDEN_CHANNELS=(32)  #64
FNO_HIDDEN_LAYERS=(3)

# RESNET parameters
RESNET_HIDDEN_CHANNELS=(32 64)
RESNET_HIDDEN_LAYERS=(2 4)

# Function to modify YAML using sed
modify_config() {
    local config_file="$1"
    local train_size="$2"
    local spatial_resolution="$3"
    local model="$4"
    local modes="$5"
    local hidden_channels="$6"
    local hidden_layers="$7"
    local levels="$8"
    local rank="$9"
    local outer_rank="${10}"
    local hidden_sizes="${11}"
    local in_shape="${12}"
    local out_shape="${13}"
    local n="${14}"
    local branch_hidden_dim="${15}"
    local trunk_hidden_dim="${16}"
    local embedding_dim="${17}"
    local s="${18}"
    local embed_dim="${19}"
    local depth="${20}"
    
    # Copy base config
    cp "$BASE_CONFIG" "$config_file"
    
    # Modify basic parameters
    sed -i "s/^train_size:.*/train_size: $train_size/" "$config_file"
    sed -i "s/^spatial_resolution:.*/spatial_resolution: $spatial_resolution/" "$config_file"
    sed -i "s/^model:.*/model: \"$model\"/" "$config_file"
    
    # Modify model-specific parameters if provided
    if [ -n "$modes" ]; then
        sed -i "s/^modes:.*/modes: $modes/" "$config_file"
    fi
    
    if [ -n "$hidden_channels" ]; then
        sed -i "s/^hidden_channels:.*/hidden_channels: $hidden_channels/" "$config_file"
    fi
    
    if [ -n "$hidden_layers" ]; then
        sed -i "s/^hidden_layers:.*/hidden_layers: $hidden_layers/" "$config_file"
    fi
    
    if [ -n "$levels" ]; then
        sed -i "s/^levels:.*/levels: $levels/" "$config_file"
    fi
    
    if [ -n "$rank" ]; then
        sed -i "s/^rank:.*/rank: $rank/" "$config_file"
    fi

    if [ -n "$outer_rank" ]; then
        sed -i "s/^outer_rank:.*/outer_rank: $outer_rank/" "$config_file"
    fi
    
    if [ -n "$hidden_sizes" ]; then
        sed -i "s/^hidden_sizes:.*/hidden_sizes: $hidden_sizes/" "$config_file"
    fi
    
    if [ -n "$in_shape" ]; then
        sed -i "s/^in_shape:.*/in_shape: $in_shape/" "$config_file"
    fi
    
    if [ -n "$out_shape" ]; then
        sed -i "s/^out_shape:.*/out_shape: $out_shape/" "$config_file"
    fi
    
    if [ -n "$n" ]; then
        sed -i "s/^n:.*/n: $n/" "$config_file"
    fi
    
    if [ -n "$branch_hidden_dim" ]; then
        sed -i "s/^branch_hidden_dim:.*/branch_hidden_dim: $branch_hidden_dim/" "$config_file"
    fi
    
    if [ -n "$trunk_hidden_dim" ]; then
        sed -i "s/^trunk_hidden_dim:.*/trunk_hidden_dim: $trunk_hidden_dim/" "$config_file"
    fi
    
    if [ -n "$embedding_dim" ]; then
        sed -i "s/^embedding_dim:.*/embedding_dim: $embedding_dim/" "$config_file"
    fi
    
    if [ -n "$s" ]; then
        sed -i "s/^s:.*/s: $s/" "$config_file"
    fi
    
    if [ -n "$embed_dim" ]; then
        sed -i "s/^embed_dim:.*/embed_dim: $embed_dim/" "$config_file"
    fi
    
    if [ -n "$depth" ]; then
        sed -i "s/^depth:.*/depth: $depth/" "$config_file"
    fi
    
    
    # HSS_MLP specific parameters - these need to be updated based on the model
    if [ "$model" == "hss_mlp_2d_2dout" ]; then
        # Update input_size and output_size to match in_shape and out_shape
        if [ -n "$in_shape" ]; then
            sed -i "s/^input_size:.*/input_size: $in_shape/" "$config_file"
        fi
        if [ -n "$out_shape" ]; then
            sed -i "s/^output_size:.*/output_size: $out_shape/" "$config_file"
        fi
        if [ -n "$outer_rank" ]; then
            sed -i "s/^outer_rank:.*/outer_rank: $outer_rank/" "$config_file"
        fi
    fi
}

# Check if base config exists
if [ ! -f "$BASE_CONFIG" ]; then
    echo "ERROR: Base config file not found: $BASE_CONFIG"
    exit 1
fi

# Create config directory if it doesn't exist
mkdir -p "$(dirname "$TEMP_CONFIG")"

echo "Starting parameter sweep..."

for spatial_resolution in "${SPATIAL_RESOLUTIONS[@]}"; do
    echo "=== Processing spatial_resolution=$spatial_resolution ==="
    
    for train_size in "${TRAIN_SIZES[@]}"; do
        echo "--- Processing train_size=$train_size ---"

        # FNO sweep
        for mode in "${FNO_MODES[@]}"; do
            for hidden_channel in "${FNO_HIDDEN_CHANNELS[@]}"; do
                for hidden_layer in "${FNO_HIDDEN_LAYERS[@]}"; do
                    # Parameters: config_file train_size spatial_resolution model modes hidden_channels hidden_layers levels rank hidden_sizes in_shape out_shape n branch_hidden_dim trunk_hidden_dim embedding_dim s embed_dim depth
                    modify_config "$TEMP_CONFIG" "$train_size" "$spatial_resolution" "fno_2d" "$mode" "$hidden_channel" "$hidden_layer" "" "" "" "" "" "" "" "" "" "" "" "" ""
                    echo "Running FNO with train_size=$train_size, spatial_resolution=$spatial_resolution, modes=$mode, hidden_channels=$hidden_channel, hidden_layers=$hidden_layer"
                    python "$MAIN_SCRIPT" --config "$TEMP_CONFIG"
                done
            done
        done


         # HSS_MLP sweep (FIXED parameter ordering)
       for outer_rank in 8 16; do
           for levels in 2; do
               for rank in 8 16; do
                   for hidden_sizes in "[$(($spatial_resolution)),$(($spatial_resolution)),$(($spatial_resolution))]" "[$(($spatial_resolution)),$(($spatial_resolution)),$(($spatial_resolution)),$(($spatial_resolution))]"; do
                        Set dimensions based on spatial resolution
                       in_shape=$spatial_resolution
                       out_shape=$spatial_resolution
                    
                        Parameters in correct order: config_file train_size spatial_resolution model modes hidden_channels hidden_layers levels rank hidden_sizes in_shape out_shape n branch_hidden_dim trunk_hidden_dim embedding_dim s embed_dim depth
                       modify_config "$TEMP_CONFIG" "$train_size" "$spatial_resolution" "hss_mlp_2d_2dout" "" "" "" "$levels" "$rank" "$outer_rank" "$hidden_sizes" "$in_shape" "$out_shape" "" "" "" "" "" "" "" 
                       echo "Running HSS_MLP with train_size=$train_size, spatial_resolution=$spatial_resolution, levels=$levels, rank=$rank, in_shape=$in_shape, out_shape=$out_shape, hidden_sizes=$hidden_sizes" , outer_rank=$outer_rank
                       python "$MAIN_SCRIPT" --config "$TEMP_CONFIG"
                   done
               done
           done
       done

        # MLP sweep
        # for hidden_channel in "${MLP_HIDDEN_CHANNELS[@]}"; do
        #     for hidden_layer in "${MLP_HIDDEN_LAYERS[@]}"; do
        #         # Parameters: config_file train_size spatial_resolution model modes hidden_channels hidden_layers levels rank hidden_sizes in_shape out_shape n branch_hidden_dim trunk_hidden_dim embedding_dim s embed_dim depth
        #         modify_config "$TEMP_CONFIG" "$train_size" "$spatial_resolution" "mlp_2d" "" "$hidden_channel" "$hidden_layer" "" "" "" "" "" "" "" "" "" "" "" "" ""
        #         echo "Running MLP with train_size=$train_size, spatial_resolution=$spatial_resolution, hidden_channels=$hidden_channel, hidden_layers=$hidden_layer"
        #         python "$MAIN_SCRIPT" --config "$TEMP_CONFIG"
        #     done
        # done

        # RESNET sweep
        for hidden_channel in "${RESNET_HIDDEN_CHANNELS[@]}"; do
            for hidden_layer in "${RESNET_HIDDEN_LAYERS[@]}"; do
                # Parameters: config_file train_size spatial_resolution model modes hidden_channels hidden_layers levels rank hidden_sizes in_shape out_shape n branch_hidden_dim trunk_hidden_dim embedding_dim s embed_dim depth
                modify_config "$TEMP_CONFIG" "$train_size" "$spatial_resolution" "resnet_2d" "" "$hidden_channel" "$hidden_layer" "" "" "" "" "" "" "" "" "" "" "" "" ""
                echo "Running RESNET with train_size=$train_size, spatial_resolution=$spatial_resolution, hidden_channels=$hidden_channel, hidden_layers=$hidden_layer"
                python "$MAIN_SCRIPT" --config "$TEMP_CONFIG"
            done
        done

        # DEEPONET sweep
        for dim in 32 64 128; do
            n=$spatial_resolution
            # Parameters: config_file train_size spatial_resolution model modes hidden_channels hidden_layers levels rank hidden_sizes in_shape out_shape n branch_hidden_dim trunk_hidden_dim embedding_dim s embed_dim depth
            modify_config "$TEMP_CONFIG" "$train_size" "$spatial_resolution" "deeponet_2d_2dout" "" "" "" "" "" "" "" "" "" "$n" "$dim" "$dim" "$dim" "" "" "" 
            echo "Running DEEPONET with train_size=$train_size, spatial_resolution=$spatial_resolution, n=$n, branch_hidden_dim=$dim, trunk_hidden_dim=$dim, embedding_dim=$dim"
            python "$MAIN_SCRIPT" --config "$TEMP_CONFIG"
        done

        # GREENLEARNING sweep
        # for embed_dim in 64 128; do
        #     for depth in 2 4 6; do
        #         s=$spatial_resolution
        #         # Parameters: config_file train_size spatial_resolution model modes hidden_channels hidden_layers levels rank hidden_sizes in_shape out_shape n branch_hidden_dim trunk_hidden_dim embedding_dim s embed_dim depth
        #         modify_config "$TEMP_CONFIG" "$train_size" "$spatial_resolution" "greenlearning_2d" "" "" "" "" "" "" "" "" "" "" "" "" "" "$s" "$embed_dim" "$depth" 
        #         echo "Running GREENLEARNING with train_size=$train_size, spatial_resolution=$spatial_resolution, s=$s, embed_dim=$embed_dim, depth=$depth"
        #         python "$MAIN_SCRIPT" --config "$TEMP_CONFIG"
        #     done
        # done

        echo "Completed train_size=$train_size"
    done
    echo "Completed spatial_resolution=$spatial_resolution"
done

# Clean up
if [ -f "$TEMP_CONFIG" ]; then
    rm "$TEMP_CONFIG"
    echo "Cleaned up temporary config file"
fi

echo "Parameter sweep complete!"
