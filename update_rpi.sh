#!/bin/bash
cd $(dirname "$0")
set -e
cd ..
rsync -avz --exclude speakers/filter/filter_cache.db speakers speakers.axiak.net:~/
