#!/usr/bin/env bash

isort src/*.py
black -l 79 src/*.py
