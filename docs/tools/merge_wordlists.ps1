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
# Usage: .\merge_wordlists.ps1 file1.txt file2.txt [file3.txt ...]

param(
    [Parameter(Mandatory=$true, ValueFromRemainingArguments=$true)]
    [string[]]$InputFiles,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputFile = ".\.wordlist_new.txt"
)

# Check if any input files were provided
if ($InputFiles.Count -eq 0) {
    Write-Error "No input files specified. Please provide at least one file path."
    exit 1
}

# Verify all input files exist
foreach ($file in $InputFiles) {
    if (-not (Test-Path $file)) {
        Write-Error "File not found: $file"
        exit 1
    }
}

# Merge, sort, and save
Get-Content $InputFiles | Sort-Object { $_ } -Culture "en-US" | Set-Content $OutputFile -Encoding UTF8

Write-Host "Successfully merged $($InputFiles.Count) file(s) into $OutputFile"
