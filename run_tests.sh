#!/bin/sh
# Run pytest with "src" as working directory
BASE_DIR=`pwd`
(
    cd "$BASE_DIR/src" && 
    python3 -m pytest "$BASE_DIR/src" "$BASE_DIR/tests"
)