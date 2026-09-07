#!/bin/sh
# llm-browser-testkit container entrypoint.
#
# The image is used in two ways:
#
# 1. Job container (GitHub Actions / Forgejo act_runner `container.image`):
#    the runner boots the image and `docker exec`s each job step into it.
#    The container's main process must therefore STAY ALIVE — with no
#    arguments we run `tail -f /dev/null`, the standard way to keep a
#    job container running. Without this, the container exits instantly
#    and the runner reports the job container as broken.
#
# 2. One-shot CLI use: `docker run <image> run /scenario.toml ...` forwards
#    its arguments to the harness, so the image doubles as a regular CLI
#    wrapper. `docker run <image> sh` drops into a debugging shell.
#
# The version is printed on every container start so CI logs always show
# which image (and therefore which harness) is being run.

set -eu

VERSION=$(llm-browser-testkit version 2>/dev/null) || VERSION="unknown"
printf 'llm-browser-testkit %s (job container ready)\n' "$VERSION"

if [ "$#" -eq 0 ]; then
    exec tail -f /dev/null
fi

case "$1" in
    sh | bash | ash | zsh) exec "$@" ;;
    *) exec llm-browser-testkit "$@" ;;
esac