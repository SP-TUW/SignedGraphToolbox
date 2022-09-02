#!/bin/sh

echo $HOSTNAME
cd ~/GSPSimulator/GSP/
echo $(pwd)
python3 -m $*
ls
