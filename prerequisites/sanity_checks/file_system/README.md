# File System Sanity Checks

This directory contains a set of self-contained sanity-check scripts used to verify user workspace access and dataset
availability on Ibex.
You can run each script individually, or run all checks at once using the master script.

These checks are intended for lightweight validation before workshops, training runs, or automated job setups.

## Contents

| Script                  | Purpose                                                                                       |
|-------------------------|-----------------------------------------------------------------------------------------------|
| `user_space_access.sh`  | Verifies the user’s `/ibex/user/$USER` directory: path, permissions, write/read/delete tests. |

## Running Checks

To run every filesystem test at once:

```commandline
bash user_space_access.sh
```

### This script:

Validates your Ibex personal directory:

- This script performs:

- Path existence check

- RWX permission check

- Write test (creates file)

- Read test

- Delete test

- Summary with success/failure indicators

## Expected Output


```commandline
[✓] Path exists: /ibex/user/<username>
[✓] Permissions OK (rwx)
[✓] Write test passed
[✓] Read test passed
[✓] Delete test passed
/ibex/user/<username> is accessible.
```
