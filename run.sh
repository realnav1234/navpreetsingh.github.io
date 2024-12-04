#!/bin/bash

arg1=$1

if [ "$arg1" == "prod" ]; then
    # Run jekyll with local configuration
    jekyll serve
else
    # Run jekyll without specifying a config
    jekyll serve --config _config-local.yml
fi