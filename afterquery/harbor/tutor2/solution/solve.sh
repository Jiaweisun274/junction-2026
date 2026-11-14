#!/bin/sh
set -eu
# Reference run: trained PPO policy (observation-only). The literature policy is in reference_agent.py.
exec python /solution/ppo_agent.py
