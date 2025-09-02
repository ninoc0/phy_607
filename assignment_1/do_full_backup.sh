#!/bin/sh

# making the script location independent, moves user to this file when ran
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# naming the source and backup directories
SRC_DIR="src"
BACKUP_DIR="backups"

# confirms a source directory exists
if [ ! -d "$SRC_DIR" ]; then
  echo "No Source directory exists" >&2
  exit 1 # error exit
fi

# creates backups if it doesnt exist already
mkdir -p "$BACKUP_DIR"

# saves the time stamp
ts="$(date '+%Y%m%d_%H%M%S')"
# where the backup writes to
archive="$BACKUP_DIR/src-$ts.tar.gz"

# something mac needs because itll include some meta data and get a weird error
export COPYFILE_DISABLE=1

# creates the backup of the src directory
tar -czf "$archive" -C "$SCRIPT_DIR" "$SRC_DIR" || {
  echo "Backup failed creating $archive" >&2
  exit 1
}

echo "Created backup: $archive"
# prints permissions info
ls -lh "$archive"

# ensuring only 3 backups save
count=0
for f in $(ls -t "$BACKUP_DIR"/src-*.tar.gz 2>/dev/null); do
  count=$((count + 1))
  if [ "$count" -gt 3 ]; then
    rm -f -- "$f" && echo "Removed old backup: $f"
  fi
done
