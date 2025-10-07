#!/usr/bin/env bash
# Create and activate a venv and install requirements
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt