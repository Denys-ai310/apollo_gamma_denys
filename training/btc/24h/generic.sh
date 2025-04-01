#!/bin/bash

# Change to the directory of the script
cd "$(dirname "$0")"

# Run the Python script with the provided arguments
/workspace/.heroku/python/bin/python "$1" "$2"