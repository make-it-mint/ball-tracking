#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#create venv if it does not exist
test -d $SCRIPT_DIR/.OpenCV_venv \
    || { \
        C:/Users/Kevin/AppData/Local/Programs/Python/Python39/Python.exe -m venv $SCRIPT_DIR/.OpenCV_venv \
        && source $SCRIPT_DIR/.OpenCV_venv/Scripts/activate \
        && $SCRIPT_DIR/.OpenCV_venv/Scripts/Python.exe -m pip install --upgrade pip \
        && pip install -r $SCRIPT_DIR/requirements.txt; \
    }

#Check all requirements are met and run
source $SCRIPT_DIR/.OpenCV_venv/Scripts/activate \
    && $SCRIPT_DIR/.OpenCV_venv/Scripts/Python.exe -m pip install --upgrade pip \
    && pip install -r $SCRIPT_DIR/requirements.txt \
    && python $SCRIPT_DIR/fps.py --display 1 --num-frames 100