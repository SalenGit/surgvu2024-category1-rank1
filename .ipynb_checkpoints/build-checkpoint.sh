#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
rm output/*
docker build -t surgtoolloc_det "$SCRIPTPATH"
