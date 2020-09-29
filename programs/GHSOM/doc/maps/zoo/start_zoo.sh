#!/bin/bash

MAP=$(basename $0 .sh)

$(dirname $0)/../startmap.sh ${MAP##"start_"}