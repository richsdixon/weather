#!/bin/bash
set -e # Exit immediately if a command fails

cd ~/git/weather

# Run Python script
python3 wunderground.py

# Force-stage wunderground.html (bypasses potential gitignore rules)
git add -f wunderground.html

# Commit changes (if any)
git commit -m "Automated update" || echo "No changes to commit"

# Push explicitly to origin and your primary branch (e.g., main)
git push origin main