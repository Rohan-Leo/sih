#!/usr/bin/env bash
# Download the synchronised phone + vehicle recordings of IO-VNBD (~410 MB) into data/raw/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw
TMP=$(mktemp -d)
git clone --depth 1 --filter=blob:limit=1k https://github.com/onyekpeu/IO-VNBD "$TMP/iovnbd"
cd "$TMP/iovnbd"
git ls-tree -r --name-only HEAD | grep "^Synchronised V abd S datasets/Categorised" | grep -i "\.csv$" | while IFS= read -r f; do
  rel="${f#Synchronised V abd S datasets/Categorised IOVNB Dataset/}"
  out="$OLDPWD/data/raw/$rel"
  [ -s "$out" ] && continue
  mkdir -p "$(dirname "$out")"
  url="https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$f")"
  curl -sSL --retry 3 -o "$out" "$url"
  echo "got $rel"
done
rm -rf "$TMP"
