#!/bin/bash

ROOT_DIR="souffle"
MAIN_DIR="souffle/results"
ERROR_LOG="souffle_error_log.txt"

if [ ! -d "$MAIN_DIR" ]; then
  echo "Main dir $MAIN_DIR is not exist, please check"
  exit 1
fi

echo "Error log for running souffle engine: " > "$ERROR_LOG"

for subfolder in "$MAIN_DIR"/*; do
  if [ -d "$subfolder" ]; then
    echo "Processing: $subfolder"

    FACTS_DIR="$subfolder/facts"
    DATALOG_FILE="$ROOT_DIR/syscall.dl"
    OUTPUT_DIR="$subfolder/output"

    if [ ! -d "$FACTS_DIR" ]; then
      echo "The subfolder $subfolder dose not contain facts "
      continue
    fi

    if [ ! -f "$DATALOG_FILE" ]; then
      echo "The folder $ROOT_DIR dose not contain Datalog file syscall.dl"
      continue
    fi

    if [ ! -d "$OUTPUT_DIR" ]; then
      mkdir -p "$OUTPUT_DIR"
    fi

    echo "Performing Souffle inference ..."
    souffle -F "$FACTS_DIR" -D "$OUTPUT_DIR" "$DATALOG_FILE"
    if [ $? -eq 0 ]; then
      echo "Subfolder $subfolder success, check $OUTPUT_DIR。"
    else
      echo "Subfolder $subfolder failed. "
      echo "$subfolder: Inference Error" >> "$ERROR_LOG"
    fi
  fi
done

echo "finished."
