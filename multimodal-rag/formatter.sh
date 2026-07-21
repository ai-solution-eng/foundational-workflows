#! /usr/bin/bash
for file in './src/multimodal_rag/' 'tests/'; do
    echo $file
    black --line-length=120 $file
    echo ''
    flake8 $file --max-line-length=120 --extend-ignore=E203
    echo ''
    mypy --python-version=3.14 --ignore-missing-imports $file
done
