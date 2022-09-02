#!/bin/sh

echo $HOSTNAME
cd ~/GSPSimulator/SignedGraphToolboxPrivate/
echo $(pwd)
python3 -m $*
ls
