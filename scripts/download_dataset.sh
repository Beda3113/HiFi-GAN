#!/bin/bash
# Download RUSLAN dataset

echo "Downloading RUSLAN dataset..."

# Create data directory
mkdir -p data/ruslan
cd data/ruslan

# Download from source (replace with actual URL)
wget https://ruslan-corpus.github.io/ruslan.zip
unzip ruslan.zip

# Create file lists
find . -name "*.wav" | shuf > all_files.txt
head -n 100 all_files.txt > validation.txt
tail -n +101 all_files.txt > training.txt

echo "Dataset downloaded and prepared"