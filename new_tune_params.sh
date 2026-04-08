#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <output_folder> [dataset]"
  exit 1
fi

OUTPUT_DIR="$(realpath "$1")"
DATASET_ARG="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TUNER_WORKDIR=$(realpath "$TUNER_WORKDIR")

if [ -n "$DATASET_ARG" ]; then
  export TUNER_DATASET="$DATASET_ARG"
  echo "Using dataset: $TUNER_DATASET"
fi

if [ -d "$OUTPUT_DIR" ]; then
  read -p "Directory '$OUTPUT_DIR' already exists. Do you want to continue? [y/N] " response
  if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Aborting."
    exit 1
  fi
else
  mkdir -p "$OUTPUT_DIR"
fi

declare -A trials_map

trials_map["mergerSectorRefit"]="4 2"
trials_map["mergerTrackFit"]="4 2"
trials_map["mergerCollect"]="4 2"
trials_map["multikernel"]="4 2"
trials_map["clusterizer"]="4 2"
trials_map["compressionStep1unattached"]="4 2"
trials_map["tracklet"]="4 2"

: "${TUNE_SPACE_DIR:=tune_spaces}"

for yaml_file in "${!trials_map[@]}"; do
  IFS=' ' read -r trials startup <<< "${trials_map[$yaml_file]}"

  export TUNE_SPACE_PATH="${TUNE_SPACE_DIR}/${yaml_file}.yaml"
  export TUNE_SPACE_NAME="$yaml_file"

  python run_optuna.py \
    --output "$OUTPUT_DIR/${yaml_file}_tuning" \
    --trials "$trials" \
    --startup "$startup"
done