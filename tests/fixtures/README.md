# Test Fixtures

This directory contains sample FIT files for testing purposes.

## Files

- `sample_activity.fit` - Sample activity file with GPS and sensor data
- `sample_workout.fit` - Sample structured workout with intervals

## Usage

These files are used by the test suite to verify FIT file processing functionality.

To add new test files:

1. Place the `.fit` file in this directory
2. Update test cases to reference the new file
3. Ensure the file contains the expected message types for your tests

## Note

Due to the binary nature of FIT files, actual sample files would need to be generated
or obtained from real devices. For development, you can use mock data in the tests
or create minimal FIT files using the FIT SDK.
