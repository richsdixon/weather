#!/bin/bash
cd ~/git/weather
# Run your Python script to generate the .html and .png files
python3 download.py
python3 gusts.py
# Package all new files and push them to GitHub Pages
git add .
git commit -m "Automated update"
git push