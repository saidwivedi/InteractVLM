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

# Prompt for credentials
echo -e "\nYou need to register at https://interactvlm.is.tue.mpg.de"
read -p "Username:" username
read -sp "Password:" password
echo

username=$(urle $username)
password=$(urle $password)

mkdir -p ./trained_models

# Download and unzip the first two zip files by default
download_and_unzip 'https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=data.zip' './'
download_and_unzip 'https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=interactvlm-3d-hcontact-damon.zip'
download_and_unzip 'https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=interactvlm-3d-oafford-lemon-piad.zip'