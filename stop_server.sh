#!/bin/bash

SESSION_NAME="senior_project"

# Kill the tmux session
tmux kill-session -t $SESSION_NAME
echo "Stopped FastAPI and recognition worker."
