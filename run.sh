#!/bin/bash

# Configuration
PORT=8000
APP_SCRIPT="app_sa.py"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Stock Advisor...${NC}"

# Check if port is in use
echo "Checking port $PORT..."
PIDS=$(lsof -ti :$PORT)

if [ ! -z "$PIDS" ]; then
    # Flatten newlines to spaces
    PIDS_FLAT=$(echo "$PIDS" | tr '\n' ' ')
    echo -e "${YELLOW}Port $PORT is currently in use by PID(s): $PIDS_FLAT${NC}"
    
    # Kill the processes
    echo -e "${RED}Killing process(es) ($PIDS_FLAT) to free up port $PORT...${NC}"
    # Use xargs to kill all PIDs
    echo "$PIDS" | xargs kill -9
    sleep 1
    
    # Verify it's gone
    if lsof -i :$PORT > /dev/null; then
        echo -e "${RED}Failed to kill process on port $PORT. Please check manually.${NC}"
        exit 1
    else
        echo -e "${GREEN}Port $PORT is now free.${NC}"
    fi
else
    echo -e "${GREEN}Port $PORT is free.${NC}"
fi

# Check for venv
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo -e "${YELLOW}Warning: 'venv' directory not found. Assuming dependencies are installed globally or in another environment.${NC}"
fi

# Run the app
echo -e "${GREEN}Launching Chainlit app...${NC}"
if [ -n "$APP_ENV" ]; then
   echo "Environment: $APP_ENV"
fi
chainlit run $APP_SCRIPT --port $PORT
