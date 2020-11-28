#!/usr/bin/env bash
cd ~/Learning/synergy/
source venv/bin/activate
cd src/
mlflow experiments csv -x 0 -o results.csv