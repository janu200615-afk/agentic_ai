# Agentic AI for Farmers — Polished Decentralized Prototype

This is a hackathon-ready prototype demonstrating an *agentic*, privacy-first, decentralized AI system for farmers.
It includes a simple aggregator server, multiple local agents that train on synthetic private data, a Streamlit UI for farmer interaction,
and example secure-aggregation/differential-privacy helpers.

## Quick start (local demo)

1. Create a virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate   # mac/linux
# venv\Scripts\activate  # windows
pip install -r requirements.txt
```

2. Start the aggregator server:
```
python server.py
```

3. In separate terminals run one or more agents (or simulate):
```
python agent.py --id 1 --simulate
python agent.py --id 2 --simulate
```

4. Run the Streamlit UI (optional):
```
streamlit run app_streamlit.py
```

5. (Optional) Run smoke tests:
```
python ci_test.py
```

---

Files included:
- server.py           : Aggregator API (Flask)
- agent.py            : Local agent simulation and client
- model.py            : PyTorch model helpers
- utils.py            : Synthetic data generation + helpers
- secure_utils.py     : AES-GCM encryption helpers + simple DP
- app_streamlit.py    : Minimal Streamlit UI for farmer interaction
- ci_test.py          : Simple smoke test script
- requirements.txt
- run_demo.sh         : Convenience script to run the demo locally

If you want changes (Flower integration, IPFS pubsub, or a recorded demo script), reply and I'll update.


## Flower federated learning

Run a Flower server:
```
python fl_server.py
```
Start multiple Flower clients:
```
python fl_client.py --id 1
python fl_client.py --id 2
```
Flower server runs on port 8080 by default in this demo.

## Audio transcription (demo)

The Streamlit UI accepts an audio file upload (wav/mp3) and attempts to transcribe it using the Google Web Speech API via the `speech_recognition` library. This requires internet and `ffmpeg` for mp3->wav conversion. If transcription fails, use the sidebar to manually type the farmer's query.


## Sample inputs

Use `sample_inputs.json` or `sample_inputs.csv` in the repo to quickly test the app. In Streamlit sidebar, click 'Load sample JSON input' or 'Load sample CSV input' to load test cases.

## Notes about images

The ZIP includes offline, compressed placeholder images for 22 crops with multilingual labels. They are placeholders generated for the demo; if you want real royalty-free photos, I can replace them in a follow-up (requires downloading and increases ZIP size).
