#!/bin/bash

# Run the Tesla-inspired dashboard
echo "Starting Tesla Analytics Dashboard..."
echo "======================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the dashboard
streamlit run src/dashboard/tesla_dashboard.py --server.port 8502 --server.headless true

echo "Dashboard is running on http://localhost:8502"