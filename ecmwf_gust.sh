#!/bin/bash
cd ~/git/weather
# Run your Python script to generate the .html and .png files
python3 your_script.py
# Package all new files and push them to GitHub Pages
git add .
git commit -m "Automated update"
git push