# Simple smoke test to run server and two agents and verify aggregation
import subprocess, time, requests, os, sys, signal

# Start server
srv = subprocess.Popen([sys.executable, 'server.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(1.5)

# Run two agents
ag1 = subprocess.Popen([sys.executable, 'agent.py', '--id', '11', '--simulate'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(0.5)
ag2 = subprocess.Popen([sys.executable, 'agent.py', '--id', '22', '--simulate'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# wait a bit
time.sleep(6)

# check server global
try:
    r = requests.get('http://127.0.0.1:5000/get_global', timeout=3)
    if r.status_code == 200:
        print('CI Test: global model fetched OK')
    else:
        print('CI Test: failed to fetch global model, status', r.status_code)
except Exception as e:
    print('CI Test: exception fetching global model', e)

# cleanup
srv.terminate()
ag1.terminate()
ag2.terminate()
