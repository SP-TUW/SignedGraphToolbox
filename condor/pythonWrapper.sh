#!/bin/sh

echo $HOSTNAME
echo "$USER"
echo $(pwd)
cd ~/GSPSimulator/SignedGraphToolboxPrivate/
echo $(pwd)
ls data/wiki_editor/
ls data/wiki_elec/
ls data/wiki_rfa/
python3 -m $*
ls
