#!/bin/bash

PROGRAM="$1"

if [[ -z "$PROGRAM" ]]; then
  echo "Usage: $0 <program_name>"
  exit 1
fi

# Starting value
VALUE=1000

# Loop 10 times
for ((i = 1; i <= 10; i++)); do
    echo "Running $PROGRAM with value $VALUE"
    ./$PROGRAM "$VALUE" >> "$(basename "$PROGRAM").log"
    VALUE=$((VALUE + 1000))
done

