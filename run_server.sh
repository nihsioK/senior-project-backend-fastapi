#!/bin/bash

SESSION_NAME="senior_project"

# Check if the tmux session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    # Create a new tmux session, but don't attach to it
    tmux new-session -d -s $SESSION_NAME

    # Start FastAPI server in the first pane
    tmux rename-window -t $SESSION_NAME "FastAPI"
    tmux send-keys -t $SESSION_NAME "source .venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8080" C-m

    # Split the window and run the recognition worker in the second pane
    tmux split-window -h -t $SESSION_NAME
    tmux send-keys -t $SESSION_NAME "source .venv/bin/activate && python app/workers/recognition_worker.py" C-m

    # Select the first pane (FastAPI logs) as the default active pane
    tmux select-pane -t $SESSION_NAME:0.0
fi

# Attach to the tmux session
tmux attach -t $SESSION_NAME
