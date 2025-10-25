#!/bin/sh

# return the number of physical cores without relying on the existence of lscpu
grep -E 'core id|physical id' /proc/cpuinfo | tr -d '\n' | sed 's;physical;\nphysical;g' | grep -v '^$' | sort -u | wc -l
