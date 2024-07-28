#!/bin/bash

# Ensure the correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <number_of_threads> <filename_prefix>"
    exit 1
fi

num_threads=$1
filename_prefix=$2

# Define the lists of n and m values
n_values=(5 6)  # You can modify this list
m_values=(2)  # You can modify this list

# Iterate through n and m values
for n in "${n_values[@]}"; do
    filename="${filename_prefix}_${n}n"

    # Ensure the file exists and is 2000 bytes
    if [ ! -f "$filename" ]; then
        echo "File not found: $filename"
        continue
    fi

    # Calculate the number of bytes per thread
    bytes_per_thread=$((2000 / num_threads))
    remaining_bytes=$((2000 % num_threads))

    for m in "${m_values[@]}"; do
        for (( thread=0; thread<num_threads; thread++ )); do
            start_point=$((thread * bytes_per_thread))
            if [ "$thread" -eq "$((num_threads - 1))" ]; then
                bytes_to_read=$((bytes_per_thread + remaining_bytes))
            else
                bytes_to_read=$bytes_per_thread
            fi

            echo "Running program with n=$n, m=$m, filename=$filename, start_point=$start_point, bytes_to_read=$bytes_to_read"
            ./out "$n" "$m" "$filename" "$start_point" "$bytes_to_read" &
        done
        wait  # Wait for all threads to finish
    done
done
