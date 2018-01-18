#!/bin/bash
mkdir -p docs
coverage-badge -f -o docs/coverage.svg
sphinx-build sphinx docs
