file='./'
black --line-length=120 $file
echo ''
flake8 $file --max-line-length=120 --extend-ignore=E203
echo ''
mypy --python-version=3.13 --ignore-missing-imports $file
