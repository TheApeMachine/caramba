#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-}"
MESSAGE="${2:-}"

if [[ -z "$TAG" || -z "$MESSAGE" ]]; then
	echo "usage: make tag <tag> <message>" >&2
	echo "  multi-word message: make tag v1.0.0 release notes here" >&2
	echo "  or: make tag TAG=v1.0.0 MESSAGE=\"release notes\"" >&2
	exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

REPOS=(
	"$ROOT/../manifesto"
	"$ROOT/../alcatraz"
	"$ROOT/../qpool"
	"$ROOT/../puter"
	"$ROOT/../hf"
	"$ROOT"
)

TAGGED_MODULES=(
	github.com/theapemachine/manifesto
	github.com/theapemachine/alcatraz
	github.com/theapemachine/qpool
	github.com/theapemachine/puter
	github.com/theapemachine/hf
	github.com/theapemachine/caramba
)

tag_repo() {
	local repo="$1"

	if [[ ! -d "$repo/.git" ]]; then
		echo "error: $repo is not a git repository" >&2
		return 1
	fi

	echo "==> tag $repo"

	(
		cd "$repo"

		git add -A

		if ! git diff --staged --quiet; then
			git commit -m "$MESSAGE"
		fi

		git push

		if git rev-parse "$TAG" >/dev/null 2>&1; then
			echo "error: tag $TAG already exists in $repo" >&2
			return 1
		fi

		git tag -a "$TAG" -m "$MESSAGE"
		git push origin "$TAG"
	)
}

module_in_go_mod() {
	local module="$1"

	grep -qE "(^|[[:space:]])${module}[[:space:]]" go.mod
}

update_go_mods() {
	local repo="$1"

	if [[ ! -f "$repo/go.mod" ]]; then
		return 0
	fi

	echo "==> go.mod $repo"

	(
		cd "$repo"

		local repo_module
		repo_module="$(sed -n 's/^module //p' go.mod | head -1)"

		local get_args=()
		for module in "${TAGGED_MODULES[@]}"; do
			if [[ "$module" == "$repo_module" ]]; then
				continue
			fi

			if module_in_go_mod "$module"; then
				get_args+=("${module}@${TAG}")
			fi
		done

		if [[ ${#get_args[@]} -eq 0 ]]; then
			echo "    no tagged internal module dependencies"
			return 0
		fi

		go get "${get_args[@]}"
		go mod tidy

		git add go.mod
		if [[ -f go.sum ]]; then
			git add go.sum
		fi

		if ! git diff --staged --quiet; then
			git commit -m "chore: bump internal modules to ${TAG}"
			git push
		fi
	)
}

for repo in "${REPOS[@]}"; do
	tag_repo "$repo"
done

echo "Tagged all repositories with $TAG"

for repo in "${REPOS[@]}"; do
	update_go_mods "$repo"
done

echo "Updated go.mod files to ${TAG}"
