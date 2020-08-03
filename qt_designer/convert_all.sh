#!/bin/sh
for uifile in *.ui
  do
  echo $uifile
  pyname=$(basename -s .ui $uifile ).py
  echo $pyname
  pyuic5 $uifile -o $pyname
done

