#!/bin/bash

# Some functions
function usage {
    echo "USAGE: $0 [-d <dataprefix>] <map>"
}

function availableMaps {
    indent="    "
    echo
    echo "Available Maps:"

    for i in $BASE_DIR/*/start_*.sh ; do
		echo "$indent$(basename $(dirname $i))"
    done
}

# $1 ... mapname (basename of the unit/wgt file)
# $2 ... filetype, eg. vec, tv, cls (extension)
# $3 ... cmd-line argument, e.g -v, --tv, -c
# $4 ... VARNAME (variable to store the path, defaults to $2 in upper case)
function findDatasetFile {
	[ -z "$1" ] && return
	[ -z "$2" ] && return
	[ -z "$3" ] && return
	local VAR=${4:-${2^^}}
	
	local vecdir="$1"
	while [[ "$(expr index "$vecdir" "-")" -gt "0" && ! -d "$VEC_DIR/$vecdir" ]]; do
		vecdir="${vecdir%-*}"
	done
	local DATA="$VEC_DIR/$vecdir"

	local vecname="$1"
	local RES=
	while [[ "$(expr index "$vecname" "-")" -gt "0" && ! -f "$DATA/$vecname.$2" ]]; do
		vecname="${vecname%-*}"
	done
	[ -f "$DATA/$vecname.$2" ] && RES="$DATA/$vecname.$2"
	
	if [ -n "$RES" ]; then
		while [[ "$(expr index "$vecname" "-")" -gt "0" && ! -f "$DATA/$vecname.$2.gz" ]]; do
			vecname="${vecname%-*}"
		done
		[ -f "$DATA/$vecname.$2.gz" ] && RES="$DATA/$vecname.$2.gz"
	fi
	
	[ -n "$RES" ] && export "$VAR=$3 $(readlink -f "$RES")"
}

# $1 ... mapname (basename of the unit/wgt file)
# $2 ... cmd-line argument, e.g -v, --tv, -c
# $3 ... VARNAME (variable to store the path, defaults DATA_PREFIX)
function findDatafilePrefix {
	[ -z "$1" ] && return
	[ -z "$2" ] && return
	local VAR=${3:-DATA_PREFIX}

	local RES=
	local vecdir="$1"
	while [[ "$(expr index "$vecdir" "-")" -gt "0" && ! -d "$VEC_DIR/$vecdir/data" ]]; do
		vecdir="${vecdir%-*}"
	done
	local DATA="$VEC_DIR/$vecdir"
	[ -d "$VEC_DIR/$vecdir/data" ] && export "$VAR=$2 $(readlink -f "$VEC_DIR/$vecdir/data")"
}


WD="$(pwd)"
BASE_DIR="$(dirname "$(readlink -f "$0")")"
cd "$WD"

if getopts "d:" flag; then
	dataprefix="-p $OPTARG"
fi
map="${!OPTIND%/}"

if [ -z "$map" ]; then
    echo "Argument missing."
    usage
    availableMaps
    exit 1
else
    # Check Map
    if  [ ! -d $BASE_DIR/$map ] || [ ! -x $BASE_DIR/$map/start_$map.sh ]; then
		echo "Unknown map: $map"
		availableMaps
		exit 2
    fi
fi

# Here we go!

somtoolbox="$(readlink -f "$BASE_DIR/../../somtoolbox.sh")"

echo "Starting $map map"

VEC_DIR="$BASE_DIR/../datasets"
findDatasetFile "$map" "tv" "--tv" "tv"
findDatasetFile "$map" "vec" "-v" "vec"
findDatasetFile "$map" "cls" "-c" "cls"
findDatasetFile "$map" "foo" "-xxx" "foo"
[ -n "$dataprefix" ] || findDatafilePrefix "$map" "-p" "dataprefix"
[ -n "$cls" ] || findDatasetFile "$map" "clsinfo" "-c" "cls"
[ -n "$cls" ] || findDatasetFile "$map" "clsinf" "-c" "cls"

MAP_DIR=$BASE_DIR/$map
unit="-u $MAP_DIR/$map.unit"
wgt="-w $MAP_DIR/$map.wgt"
if  [ -f $MAP_DIR/$map.dwm ] || [ -f $MAP_DIR/$map.dwm.gz ]; then
    dwm="--dw $MAP_DIR/$map.dwm"
fi
if  [ -f $MAP_DIR/$map.map ] || [ -f $MAP_DIR/$map.map.gz ]; then
	mapfile="-m $MAP_DIR/$map.map"
fi


echo $somtoolbox SOMViewer $vec $unit $wgt $tv $cls $dwm $mapfile $dataprefix $foo
#$somtoolbox SOMViewer $vec $unit $wgt $tv $cls $dwm $mapfile $dataprefix
