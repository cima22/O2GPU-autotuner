#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <output_folder>"
  exit 1
fi

OUTPUT_DIR="$(realpath "$1")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "$OUTPUT_DIR" ]; then
  read -p "⚠️ Directory '$OUTPUT_DIR' already exists. Do you want to continue? [y/N] " response
  if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "❌ Aborting."
    exit 1
  fi
else
  mkdir -p "$OUTPUT_DIR"
fi

declare -A trials_map
trials_map["mergerSectorRefit"]="100 25"
trials_map["mergerTrackFit"]="100 25"
trials_map["mergerCollect"]="100 25"
trials_map["multikernel"]="200 50"
#trials_map["tracklet"]="400 100"
trials_map["clusterizer"]="400 100"
trials_map["compressionStep1unattached"]="100 25"

COMMON_CONFIG=$SCRIPT_DIR/"config.yaml"
TMP_CONFIG="$OUTPUT_DIR"/tmp_config.yaml
: "${TUNE_SPACE_DIR:=tune_spaces/MI50}"

for yaml_file in "${!trials_map[@]}"; do
  IFS=' ' read -r trials startup <<< "${trials_map[$yaml_file]}"

  echo "🔧 Updating config for $yaml_file with trials=$trials and startup=$startup"

  # Modify the common config with sed (backup first)
  cp "$COMMON_CONFIG" "$TMP_CONFIG"

  # Replace trials and startup values
  sed -i "s/trials: .*/trials: $trials/" $TMP_CONFIG
  sed -i "s/n_startup_trials: .*/n_startup_trials: $startup/" $TMP_CONFIG

  export TUNE_SPACE_PATH="${TUNE_SPACE_DIR}/${yaml_file}.yaml"
  echo "🚀 Running for $TUNE_SPACE_PATH"
  cd $SCRIPT_DIR
  # Run your actual command using the updated config
  o2tuner -c $TMP_CONFIG -w "$OUTPUT_DIR/${yaml_file}_tuning" -s opt1
done
