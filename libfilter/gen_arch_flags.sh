#!/bin/bash


ARCH_LIST="armv8-a armv7ve armv7e-m armv7-r armv7-m armv7-a armv7 armv6zk armv6z armv6t2 armv6s-m armv6k armv6j armv6-m armv6 armv5te armv5t armv5e armv5 armv4t armv4 armv3m armv3 armv2a armv2"

ARGS=""

if gcc -Q --help=target 2>/dev/null | grep -A3 'mfpu=' | grep -q neon; then
    ARGS="$ARGS -mfpu=neon"
fi

for ARCH in $ARCH_LIST; do
    if gcc -Q --help=target 2>/dev/null | grep -A3 'march=' | grep -q $ARCH; then
        ARGS="$ARGS -march=$ARCH"
        break
    fi
done

echo $ARGS
