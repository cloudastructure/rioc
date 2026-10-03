#!/usr/bin/env bash
# Install the GitHub CLI if the runner image does not carry it.
# Pinned version: an unpinned "latest" download would undo the point of
# pinning the actions themselves.
set -euo pipefail

if command -v gh >/dev/null 2>&1; then
  exit 0
fi

GH_VERSION="2.63.0"

# The runner pool's architecture is not guaranteed, and an amd64 binary on an
# arm64 runner fails at exec time with nothing pointing back to this step.
case "$(uname -m)" in
  x86_64) GH_ARCH="amd64" ;;
  aarch64 | arm64) GH_ARCH="arm64" ;;
  *) echo "::error::Unsupported architecture for the gh download: $(uname -m)" >&2; exit 1 ;;
esac

curl -fsSL "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_${GH_ARCH}.tar.gz" \
  -o /tmp/gh.tar.gz
mkdir -p /tmp/gh-cli
tar -xzf /tmp/gh.tar.gz -C /tmp/gh-cli --strip-components=1
echo "/tmp/gh-cli/bin" >> "$GITHUB_PATH"
