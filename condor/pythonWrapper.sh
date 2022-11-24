#!/bin/sh

echo $HOSTNAME
echo "$USER"
echo $(pwd)
cd ~/GSPSimulator/SignedGraphToolboxPrivate/
python3 -m $*
ls
