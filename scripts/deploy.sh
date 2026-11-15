#!/usr/bin/env bash
#
# Build the site locally and publish it to the gh-pages branch.
#
# This replaces the GitHub Actions deployment (.github/workflows/deploy.yml).
# GitHub Pages is configured to serve the gh-pages branch, so each run replaces
# that branch with one fresh commit containing exactly the files in public/.
#
# Usage:
#   ./scripts/deploy.sh                  # build, deploy, then wait for Pages
#   ./scripts/deploy.sh --skip-build     # deploy the existing public/ as-is
#   ./scripts/deploy.sh --no-wait        # do not poll the Pages build status
#   ./scripts/deploy.sh -m "message"     # custom commit message
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-gh-pages}"
WORKTREE=".deploy-gh-pages"
PUBLIC_DIR="public"

SKIP_BUILD=0
WAIT=1
MESSAGE="Deploy $(date '+%Y-%m-%d %H:%M:%S %z')"

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=1; shift ;;
    --no-wait) WAIT=0; shift ;;
    -m|--message)
      [[ $# -ge 2 ]] || die "-m/--message requires an argument"
      MESSAGE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

command -v git >/dev/null 2>&1 || die "git not found"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git repository"
git remote get-url "$REMOTE" >/dev/null 2>&1 || die "git remote '$REMOTE' not found"

if [[ $SKIP_BUILD -eq 0 ]]; then
  command -v hugo >/dev/null 2>&1 || die "hugo not found; install Hugo extended >= 0.157"
  hugo version | grep -q '+extended' || die "Hugo extended is required: $(hugo version)"
  info "Building with $(hugo version | awk '{print $2}')"
  hugo --minify --gc --cleanDestinationDir
else
  [[ -f "$PUBLIC_DIR/index.html" ]] || die "--skip-build was given but $PUBLIC_DIR/index.html does not exist"
  info "Skipping build; deploying the existing $PUBLIC_DIR/"
fi

# GitHub Pages runs Jekyll on branch sources unless this marker file exists.
touch "$PUBLIC_DIR/.nojekyll"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  warn "you have uncommitted tracked changes; they will be published by this deploy"
fi

cleanup() {
  if [[ -d "$WORKTREE" ]]; then
    git worktree remove --force "$WORKTREE" >/dev/null 2>&1 || rm -rf -- "$WORKTREE"
  fi
  git worktree prune >/dev/null 2>&1 || true
}
trap cleanup EXIT

info "Staging $PUBLIC_DIR/ in temporary worktree $WORKTREE"
rm -rf -- "$WORKTREE"
git worktree add --detach --force "$WORKTREE" HEAD >/dev/null
rsync -a --delete --exclude='.git' "$PUBLIC_DIR"/ "$WORKTREE"/
git -C "$WORKTREE" add -A

TREE="$(git -C "$WORKTREE" write-tree)"

# Compare with what is currently published and skip identical deploys.
OLD_TREE=""
if git fetch --quiet "$REMOTE" "$BRANCH" 2>/dev/null; then
  OLD_TREE="$(git rev-parse --verify 'FETCH_HEAD^{tree}' 2>/dev/null || true)"
fi
if [[ -n "$OLD_TREE" && "$OLD_TREE" == "$TREE" ]]; then
  info "Output is identical to $REMOTE/$BRANCH; nothing to deploy."
  exit 0
fi

# A parentless commit mirrors the old workflow's single-commit setting and keeps
# the branch from growing a long history of generated files.
COMMIT="$(git -C "$WORKTREE" commit-tree "$TREE" -m "$MESSAGE")"

info "Pushing to $REMOTE/$BRANCH"
git push --force "$REMOTE" "$COMMIT:refs/heads/$BRANCH" >/dev/null
info "Pushed $(git rev-parse --short "$COMMIT") to $BRANCH"

SLUG=""
PAGES_URL=""
if command -v gh >/dev/null 2>&1; then
  SLUG="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)"
fi

if [[ $WAIT -eq 1 && -n "$SLUG" ]]; then
  info "Waiting for GitHub Pages to build ..."
  DEPLOYED_SHA="$(git rev-parse "$COMMIT")"
  DONE=0
  for _ in $(seq 1 60); do
    LINE="$(gh api "repos/$SLUG/pages/builds/latest" --template '{{.status}} {{.commit}}' 2>/dev/null || true)"
    STATUS="${LINE%% *}"
    SHA="${LINE#* }"
    if [[ "$SHA" == "$DEPLOYED_SHA" ]]; then
      case "$STATUS" in
        built)
          DONE=1
          break ;;
        errored)
          die "Pages build failed: https://github.com/$SLUG/deployments" ;;
      esac
    fi
    sleep 5
  done
  PAGES_URL="$(gh api "repos/$SLUG/pages" --template '{{.html_url}}' 2>/dev/null || true)"
  if [[ $DONE -eq 1 ]]; then
    info "Pages build finished: ${PAGES_URL:-https://github.com/$SLUG/deployments}"
  else
    warn "Pages build still in progress; check https://github.com/$SLUG/deployments"
  fi
else
  [[ -n "$SLUG" ]] && PAGES_URL="$(gh api "repos/$SLUG/pages" --template '{{.html_url}}' 2>/dev/null || true)"
  info "Deployed. GitHub Pages will update shortly${PAGES_URL:+: $PAGES_URL}"
fi
