#!/usr/bin/env bash
# Start server, multiple agents, and print instructions for Streamlit
python server.py &
SERVER_PID=$!
sleep 1
python agent.py --id 1 --simulate &
python agent.py --id 2 --simulate &
echo "Server started. To view UI run: streamlit run app_streamlit.py"
wait
kill $SERVER_PID
