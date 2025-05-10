#!/bin/bash

run_parallel() {
  i="$1"         # Iteration
  program="$2"   # Program name
  size=$((1000 * i))
  echo "Running $program with size $size (Iteration $i)"
  ./"$program" "$size"
}

export -f run_parallel

# Program name passed as the first script argument
program="$1"

if [[ -z "$program" ]]; then
  echo "Usage: $0 <program_name>"
  exit 1
fi

# Number of iterations
iterations=10
parallel_jobs=8

# Run the function in parallel with values 1 to $iterations
seq 1 "$iterations" | parallel -j "$parallel_jobs" run_parallel {} "$program" >> "$program".log

exit 0

