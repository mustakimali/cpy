#!/usr/bin/bash

set -euo pipefail

run_test() {
    local title=$1
    local file=$2
    echo "> $title"
    uv run c.py examples/$file
}

run_test "Basic tests" "string.tc"
run_test "Fibonacci tests" "fibonacci.tc"
run_test "Simple function tests" "fns/simple.tc"
#run_test "Module and function tests" "fns/main.tc"
