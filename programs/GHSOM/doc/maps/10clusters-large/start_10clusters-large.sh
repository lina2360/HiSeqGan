#!/bin/bash

#FIXME: would be good to use the startmap.sh script, but it doesn't work when we have input and output called differently (10clusters-large.unit vs 10clusters.vec ...)

MAP="10clusters-large"

../../../somtoolbox.sh SOMViewer -u ${MAP}.unit -w ${MAP}.wgt --dw ${MAP}.dwm -v ../../datasets/10clusters.vec -t ../../datasets/10clusters.tv -c ../../datasets/10clusters.clsinf
