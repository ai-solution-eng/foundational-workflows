#! /usr/bin/bash
# Ruff handles both formatting (replaces black) and linting (replaces flake8).
# Line-length (120) and rule selection are configured in ruff.toml.
for file in './src/multimodal_rag/' 'tests/'; do
    echo $file
    ruff format $file
    echo ''
    ruff check --fix $file
    echo ''
    mypy --python-version=3.14 --ignore-missing-imports $file
done
