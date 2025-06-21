#!/bin/bash

urle () { 
    [[ "${1}" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do 
        x="${1:i:1}"
        [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
    done
    echo
}

# Function to download, unzip, and remove the zip file
download_and_unzip() {
    local url=$1
    local output_file=$(basename "$url" | sed 's/.*sfile=//')

    wget --post-data "username=$username&password=$password" "$url" -O "$output_file" --no-check-certificate --continue
    unzip $output_file
    rm $output_file
}

# Function to download, extract tar.gz, and remove the tar.gz file
download_and_extract_targz() {
    local url=$1
    local output_file=$(basename "$url" | sed 's/.*sfile=//')

    wget --post-data "username=$username&password=$password" "$url" -O "$output_file" --no-check-certificate --continue
    tar -xzf $output_file
    rm $output_file
}

# Prompt for credentials
echo -e "\nYou need to register at https://interactvlm.is.tue.mpg.de"
read -p "Username:" username
read -sp "Password:" password
echo

username=$(urle $username)
password=$(urle $password)

mkdir -p ./trained_models

# Define download URLs (using placeholder links for now)
DATA_URL='https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=data.zip'
HCONTACT_DAMON_MODEL_URL='https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=interactvlm-3d-hcontact-damon.zip'
OAFFORD_LEMON_PIAD_MODEL_URL='https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=interactvlm-3d-oafford-lemon-piad.zip'
DAMON_DATASET_URL='https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=damon.tar.gz'

# Check command line arguments
if [ $# -eq 0 ]; then
    # No arguments provided - download all files
    echo "No argument provided. Downloading all files..."
    download_and_unzip "$DATA_URL"
    download_and_unzip "$HCONTACT_DAMON_MODEL_URL"
    download_and_unzip "$OAFFORD_LEMON_PIAD_MODEL_URL"
elif [ "$1" = "damon-dataset" ]; then
    # damon-dataset argument provided - download only that file
    echo "Downloading DAMON dataset only..."
    download_and_extract_targz "$DAMON_DATASET_URL"
else
    echo "Unknown argument: $1"
    echo "Usage: $0 [damon-dataset]"
    echo "  No argument: Downloads all model files (data.zip, interactvlm-3d-hcontact-damon.zip, interactvlm-3d-oafford-lemon-piad.zip)"
    echo "  damon-dataset: Downloads only damon.tar.gz dataset"
    exit 1
fi