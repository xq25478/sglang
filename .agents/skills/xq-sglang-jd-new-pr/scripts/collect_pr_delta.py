#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
CI_BASE_ENV_VARS = (
    "JD_CI_PR_BASE_REF",
    "CI_MERGE_REQUEST_TARGET_BRANCH_NAME",
    "GITHUB_BASE_REF",
)


class CollectorError(RuntimeError):
    pass


def run_git(repo: Path, *args: str, check: bool = True) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise CollectorError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def resolve_ref(repo: Path, ref: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise CollectorError(f"cannot resolve base ref: {ref}")
    return result.stdout.strip()


def current_branch(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _resolve_ci_base(repo: Path) -> tuple[str, str] | None:
    for variable in CI_BASE_ENV_VARS:
        value = os.environ.get(variable, "").strip()
        if not value:
            continue
        for candidate in (value, f"origin/{value}"):
            try:
                return value, resolve_ref(repo, candidate)
            except CollectorError:
                continue
        raise CollectorError(
            f"cannot resolve base ref from {variable}: {value}; pass --base explicitly"
        )
    return None


def _jd_release_refs(repo: Path) -> list[tuple[str, str]]:
    output = run_git(
        repo,
        "for-each-ref",
        "--format=%(refname)%00%(objectname)",
        "refs/heads",
        "refs/remotes",
    )
    refs = []
    for raw_line in output.decode("utf-8", errors="replace").splitlines():
        if "\x00" not in raw_line:
            continue
        ref, sha = raw_line.split("\x00", 1)
        short_name = ref.rsplit("/", 1)[-1]
        if short_name.startswith("JD-v") and SHA_PATTERN.fullmatch(sha):
            refs.append((ref, sha))
    return refs


def _infer_jd_base(repo: Path, head: str) -> tuple[str, str]:
    by_sha: dict[str, list[str]] = {}
    for ref, sha in _jd_release_refs(repo):
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, head],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ancestor.returncode == 0 and sha != head:
            by_sha.setdefault(sha, []).append(ref)

    if not by_sha:
        raise CollectorError(
            "cannot infer JD base; pass an explicit base with --base"
        )

    distances = {
        sha: int(run_git(repo, "rev-list", "--count", f"{sha}..{head}").strip())
        for sha in by_sha
    }
    nearest_distance = min(distances.values())
    nearest = [sha for sha, distance in distances.items() if distance == nearest_distance]
    if len(nearest) != 1:
        labels = [sorted(by_sha[sha])[0] for sha in sorted(nearest)]
        raise CollectorError(
            "ambiguous JD base: "
            + ", ".join(labels)
            + "; pass an explicit base with --base"
        )

    sha = nearest[0]
    return sorted(by_sha[sha], key=lambda ref: (ref.startswith("refs/remotes/"), ref))[0], sha


def resolve_base(repo: Path, explicit_base: str | None, head: str) -> tuple[str, str, str]:
    if explicit_base:
        return explicit_base, resolve_ref(repo, explicit_base), "explicit"
    if current_branch(repo) is None:
        raise CollectorError("detached HEAD requires an explicit base with --base")
    ci_base = _resolve_ci_base(repo)
    if ci_base:
        requested, resolved = ci_base
        return requested, resolved, "ci-environment"
    requested, resolved = _infer_jd_base(repo, head)
    return requested, resolved, "jd-release-ancestor"


def parse_name_status(output: bytes) -> list[dict[str, str]]:
    tokens = [
        token.decode("utf-8", errors="surrogateescape")
        for token in output.split(b"\x00")
        if token
    ]
    entries = []
    index = 0
    while index < len(tokens):
        status = tokens[index]
        index += 1
        if status.startswith(("R", "C")):
            if index + 1 >= len(tokens):
                raise CollectorError("malformed rename/copy output from git")
            old_path, path = tokens[index], tokens[index + 1]
            entries.append({"status": status, "old_path": old_path, "path": path})
            index += 2
        else:
            if index >= len(tokens):
                raise CollectorError("malformed file-status output from git")
            entries.append({"status": status, "path": tokens[index]})
            index += 1
    return entries


def collect_commit_files(repo: Path, sha: str) -> list[dict[str, str]]:
    commit_and_parents = run_git(repo, "rev-list", "--parents", "-n", "1", sha).decode().split()
    if len(commit_and_parents) > 2:
        output = run_git(
            repo,
            "diff",
            "--name-status",
            "-z",
            "-M",
            commit_and_parents[1],
            sha,
        )
    else:
        output = run_git(
            repo,
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--name-status",
            "-z",
            "-M",
            sha,
        )
    return parse_name_status(output)


def collect_commits(repo: Path, merge_base: str, head: str) -> list[dict[str, object]]:
    shas = run_git(repo, "rev-list", "--reverse", f"{merge_base}..{head}").decode().split()
    commits = []
    for sha in shas:
        metadata = run_git(
            repo,
            "show",
            "-s",
            "--format=%H%x00%s%x00%an%x00%ae",
            sha,
        ).decode("utf-8", errors="replace").rstrip("\n").split("\x00")
        if len(metadata) != 4:
            raise CollectorError(f"unexpected commit metadata for {sha}")
        files = collect_commit_files(repo, sha)
        commits.append(
            {
                "sha": metadata[0],
                "subject": metadata[1],
                "author_name": metadata[2],
                "author_email": metadata[3],
                "files": files,
            }
        )
    return commits


def _literal_strings(node: ast.AST) -> list[str]:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (tuple, list, set)):
        return [item for item in value if isinstance(item, str)]
    return []


def parse_manifest(path: Path, repo_root: Path) -> dict[str, object]:
    display_path = str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)
    if not path.is_file():
        return {
            "path": display_path,
            "exists": False,
            "internal_commits": [],
            "mapped_commits": [],
        }

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    internal_commits: set[str] = set()
    mapped_commits: set[str] = set()
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == "INTERNAL_COMMITS" and value:
            internal_commits.update(_literal_strings(value))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        if function_name != "JDCase":
            continue
        for keyword in node.keywords:
            if keyword.arg == "commits":
                mapped_commits.update(_literal_strings(keyword.value))

    return {
        "path": display_path,
        "exists": True,
        "internal_commits": sorted(
            commit for commit in internal_commits if SHA_PATTERN.fullmatch(commit)
        ),
        "mapped_commits": sorted(
            commit for commit in mapped_commits if SHA_PATTERN.fullmatch(commit)
        ),
    }


def collect_delta(repo: Path, base: str | None, manifest: Path) -> dict[str, object]:
    root = Path(run_git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    head = resolve_ref(root, "HEAD")
    branch = current_branch(root)
    requested_base, resolved_base, base_source = resolve_base(root, base, head)
    merge_base = run_git(root, "merge-base", resolved_base, head).decode().strip()
    commits = collect_commits(root, merge_base, head)
    if not commits:
        raise CollectorError(
            f"comparison range is empty: {merge_base}..{head}"
        )

    manifest_path = manifest if manifest.is_absolute() else root / manifest
    manifest_data = parse_manifest(manifest_path.resolve(), root)
    mapped = set(manifest_data["mapped_commits"])
    files = parse_name_status(
        run_git(root, "diff", "--name-status", "-z", "-M", merge_base, head)
    )
    return {
        "repository": str(root),
        "branch": branch,
        "base_source": base_source,
        "requested_base": requested_base,
        "resolved_base": resolved_base,
        "merge_base": merge_base,
        "head": head,
        "commits": commits,
        "files": files,
        "manifest": manifest_data,
        "uncovered_commits": [
            commit["sha"] for commit in commits if commit["sha"] not in mapped
        ],
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect an auditable JD PR delta and manifest coverage report"
    )
    parser.add_argument("--repo", default=".", help="Git repository to inspect")
    parser.add_argument("--base", help="Explicit PR target or JD release base ref")
    parser.add_argument(
        "--manifest",
        default="test/jd-ci/jd_test_manifest.py",
        help="JD manifest path, relative to the repository root",
    )
    parser.add_argument("--output", help="Write JSON to this path instead of stdout")
    return parser.parse_args(argv)


def write_json(data: dict[str, object], output: str | None) -> None:
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if not output:
        sys.stdout.write(payload)
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        data = collect_delta(Path(args.repo).resolve(), args.base, Path(args.manifest))
        write_json(data, args.output)
    except (CollectorError, OSError, SyntaxError) as error:
        print(f"collect_pr_delta: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
