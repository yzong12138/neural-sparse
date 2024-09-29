#!/bin/bash

# List of languages
langs=('ar' 'bn' 'ko' 'de' 'es' 'fa' 'fi' 'fr' 'hi' 'id' 'ja' 'ru' 'sw' 'te' 'th' 'yo' 'zh' 'en')

# Loop through each language
for lang in "${langs[@]}"; do
    echo "Running parse_miracl.py with language: $lang"
    python parse_miracl.py "$lang" &
done