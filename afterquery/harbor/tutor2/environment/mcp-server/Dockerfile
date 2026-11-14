FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py tutor2.py lit_ref.py sweep_tutor2.py ppo_policy.py ppo_ref.npz ./

EXPOSE 8000
CMD ["python", "server.py"]
