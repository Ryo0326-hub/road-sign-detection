#!/usr/bin/env bash
n=0
while IFS= read -r u; do
  [ -z "$u" ] && continue
  n=$((n+1))
  hdr=$(curl -sL -o /dev/null -D - -r 0-0 --max-time 30 "$u" 2>/dev/null | tr -d '\r')
  name=$(printf '%s' "$hdr" | grep -i 'content-disposition' | sed -n 's/.*filename=\"\{0,1\}\([^\";]*\).*/\1/p' | tail -1)
  size=$(printf '%s' "$hdr" | grep -i 'content-range' | sed -n 's#.*/##p' | tail -1)
  if [ -n "$size" ]; then hs=$(awk -v b="$size" 'BEGIN{printf "%.2f GB", b/1073741824}'); else hs="?"; fi
  printf "%2d  %-48s %s\n" "$n" "${name:-unknown}" "$hs"
done < "${1:-urls.txt}"
