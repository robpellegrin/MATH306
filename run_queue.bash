#!#!/bin/bash

PROGRAM="$1"

if [[ -z "$PROGRAM" ]]; then
  echo "Usage: $0 <program_name>"
  exit 1
fi

# Starting value
VALUE=1000

# Loop 10 times
for ((i = 1; i <= 20; i++)); do
    total=0
    echo "Running $PROGRAM with value $VALUE"

    # Run the program 3 times, collect output and sum
    for ((j = 1; j <= 3; j++)); do
        result=$($PROGRAM "$VALUE")
        total=$((total + result))
    done

    # Calculate average
    average=$((total / 3))

    # Log the average and display it
    echo "Average output: $average"
    echo "$VALUE -> Average: $average" >> "$(basename "$PROGRAM").log"

    # Update VALUE for the next iteration
    VALUE=$((VALUE + 1000))
done

