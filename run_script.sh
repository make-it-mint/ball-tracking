#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#create venv if it does not exist
test -d $SCRIPT_DIR/.ball-tracking_venv \
    || { \
        C:/Users/Kevin/AppData/Local/Programs/Python/Python39/Python.exe -m venv $SCRIPT_DIR/.ball-tracking_venv \
        && source $SCRIPT_DIR/.ball-tracking_venv/Scripts/activate \
        && $SCRIPT_DIR/.ball-tracking_venv/Scripts/Python.exe -m pip install --upgrade pip \
        && pip install -r $SCRIPT_DIR/requirements.txt; \
    }

#Check all requirements are met and run
source $SCRIPT_DIR/.ball-tracking_venv/Scripts/activate \
    && $SCRIPT_DIR/.ball-tracking_venv/Scripts/Python.exe -m pip install --upgrade pip \
    && pip install -r $SCRIPT_DIR/requirements.txt \
    && python $SCRIPT_DIR/read_video.py 