#!/bin/bash

# This script wraps the benchmark management tool in a watch command
# to provide a live-updating dashboard of your experiments.
# Refresh interval: 120 seconds (2 minutes)

watch --interval 120 --color python manage_benchmark_status.py report "$@"
