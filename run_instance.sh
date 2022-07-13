#!/bin/bash
# prepare_instance.sh script for VNNCOMP for MNBAB: # four arguments, first is "v1", second is a benchmark category identifier string, third is path to the .onnx file and fourth is path to .vnnlib file
# Stanley Bak, Feb 2021

TOOL_NAME=MNBaB
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
run_file="$SCRIPT_DIR/src/run_instance.py"

echo "Running $TOOL_NAME for benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE' and vnnlib file '$VNNLIB_FILE' and timeout '$TIMEOUT'. Writing to '$RESULTS_FILE'"

python $run_file --benchmark $CATEGORY --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --results_path $RESULTS_FILE --timeout $TIMEOUT

exit 0
