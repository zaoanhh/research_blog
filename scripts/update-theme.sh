#!/usr/bin/env bash
#
# Update the Stack theme locally (replaces the old update-theme workflow).
# It bumps go.mod/go.sum and verifies the site still builds; commit and
# deploy afterwards with ./scripts/deploy.sh.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v hugo >/dev/null 2>&1 || die "hugo not found; install Hugo extended >= 0.157"
command -v go >/dev/null 2>&1 || die "go not found; Hugo modules need Go"

info "Updating Hugo modules"
hugo mod get -u

info "Tidying go.mod / go.sum"
hugo mod tidy

info "Verifying the site still builds"
hugo --minify --gc --cleanDestinationDir

echo
info "Review the changes with 'git diff go.mod go.sum', then commit and run ./scripts/deploy.sh"
