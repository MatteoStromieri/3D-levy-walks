#!/bin/bash

# === OAR directives ===
#OAR -l walltime=10:00:00
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err
#OAR -n fixed-surface

SOURCE_FILE="simulation.c"
LIB_FILE="func.c helpers.c"
EXECUTABLE_NAME="simulation_executable_$$"  # Executable with unique name
CONFIG_FILE="./experiments/configs/detection_time_fixed_surface.conf"

echo "Compiling $SOURCE_FILE..."
gcc "$SOURCE_FILE" $LIB_FILE -o "$EXECUTABLE_NAME" -lm
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed for $SOURCE_FILE."
    exit 1
fi
echo "Compilation successful. Executable: $EXECUTABLE_NAME"

# === Running the simulation ===
echo "Running the simulation..."
"./$EXECUTABLE_NAME" "$CONFIG_FILE"
if [ $? -ne 0 ]; then
    echo "Error: Simulation execution failed."
else
    echo "Simulation completed successfully."
fi

# === Cleanup ===
echo "Deleting the executable $EXECUTABLE_NAME..."
rm -f "./$EXECUTABLE_NAME"
if [ $? -ne 0 ]; then
    echo "Warning: Could not delete the executable $EXECUTABLE_NAME."
fi
