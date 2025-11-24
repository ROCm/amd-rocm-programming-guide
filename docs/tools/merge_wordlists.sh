#!/bin/bash
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Merge and sort multiple wordlist files
# Usage: ./merge_wordlists.sh file1.txt file2.txt [file3.txt ...]

# Default output file
OUTPUT_FILE=".wordlist_new.txt"

# Check if any input files were provided
if [ $# -eq 0 ]; then
    echo "Error: No input files specified. Please provide at least one file path." >&2
    exit 1
fi

# Verify all input files exist
for file in "$@"; do
    if [ ! -f "$file" ]; then
        echo "Error: File not found: $file" >&2
        exit 1
    fi
done

# Merge, sort, and save
cat "$@" | sort > "$OUTPUT_FILE"

echo "Successfully merged $# file(s) into $OUTPUT_FILE"
