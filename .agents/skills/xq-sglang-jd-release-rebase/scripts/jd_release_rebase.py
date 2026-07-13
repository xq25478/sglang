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


CONFLICT_EXIT_CODE = 3
CHECK_FAILED_EXIT_CODE = 4
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
CONFLICT_MARKER = re.compile(r"^(?:<<<<<<< |=======$|>>>>>>> )")
HIGH_RISK_MARKERS = (
    "pyproject.toml",
    "requirements",
    "CMakeLists.txt",
    "sgl-kernel/python/sgl_kernel/__init__.py",
    "python/sglang/srt/models/",
    "python/sglang/srt/distributed",
    "python/sglang/srt/managers/",
    "test/jd-ci/env/",
    "test/jd-ci/run_jd_ci.sh",
)
JD_CI_COMMIT_PATH_PREFIXES = (
    "test/jd-ci/",
    ".agents/skills/xq-sglang-jd-new-pr/",
    ".agents/skills/xq-sglang-jd-release-rebase/",
)


class RebaseError(RuntimeError):
    pass


def run_git(
    repo: Path,
    *args: str,
    check: bool = True,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RebaseError(f"git {' '.join(args)} failed: {detail}")
    return result


def repository_root(path: Path) -> Path:
    result = run_git(path, "rev-parse", "--show-toplevel")
    return Path(result.stdout.strip()).resolve()


def git_common_dir(repo: Path) -> Path:
    raw = run_git(repo, "rev-parse", "--git-common-dir").stdout.strip()
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def all_refs(repo: Path) -> list[tuple[str, str]]:
    output = run_git(
        repo,
        "for-each-ref",
        "--format=%(refname)%00%(objectname)%00%(*objectname)",
        "refs/heads",
        "refs/tags",
        "refs/remotes",
    ).stdout
    refs = []
    for line in output.splitlines():
        if "\x00" not in line:
            continue
        parts = line.split("\x00")
        if len(parts) != 3:
            continue
        ref, object_sha, peeled_sha = parts
        sha = peeled_sha if SHA_PATTERN.fullmatch(peeled_sha) else object_sha
        if SHA_PATTERN.fullmatch(sha):
            refs.append((ref, sha))
    return refs


def resolve_full_ref(repo: Path, ref: str) -> dict[str, str]:
    refs = all_refs(repo)
    if ref.startswith("refs/"):
        matches = [(name, sha) for name, sha in refs if name == ref]
        if not matches:
            raise RebaseError(f"cannot resolve ref: {ref}")
        return {"ref": matches[0][0], "sha": matches[0][1]}

    candidate_names = {
        f"refs/heads/{ref}",
        f"refs/tags/{ref}",
        f"refs/remotes/{ref}",
    }
    matches = [(name, sha) for name, sha in refs if name in candidate_names]
    if not matches:
        raise RebaseError(f"cannot resolve ref: {ref}; use a full ref")
    distinct_shas = {sha for _, sha in matches}
    if len(distinct_shas) != 1:
        labels = ", ".join(name for name, _ in sorted(matches))
        raise RebaseError(f"ambiguous ref {ref}: {labels}; use a full ref")
    priority = {"refs/heads/": 0, "refs/remotes/": 1, "refs/tags/": 2}
    matches.sort(
        key=lambda item: next(
            rank for prefix, rank in priority.items() if item[0].startswith(prefix)
        )
    )
    return {"ref": matches[0][0], "sha": matches[0][1]}


def infer_old_upstream(old_internal_ref: str) -> str:
    branch_name = old_internal_ref.rsplit("/", 1)[-1]
    if not branch_name.startswith("JD-v"):
        raise RebaseError(
            "cannot infer old upstream tag; pass --old-upstream explicitly"
        )
    return f"refs/tags/{branch_name.removeprefix('JD-')}"


def infer_new_internal(new_upstream_ref: str) -> str:
    tag_name = new_upstream_ref.rsplit("/", 1)[-1]
    if not tag_name.startswith("v"):
        raise RebaseError(
            "cannot infer new JD branch; pass --new-internal explicitly"
        )
    return f"JD-{tag_name}"


def normalize_new_internal(name: str) -> str:
    if name.startswith("refs/heads/"):
        name = name.removeprefix("refs/heads/")
    if not name or name.startswith("refs/") or name.endswith("/HEAD"):
        raise RebaseError(f"invalid new internal branch name: {name}")
    return name


def new_internal_exists(repo: Path, name: str) -> list[str]:
    local = f"refs/heads/{name}"
    matches = [ref for ref, _ in all_refs(repo) if ref == local]
    matches.extend(
        ref
        for ref, _ in all_refs(repo)
        if ref.startswith("refs/remotes/") and ref.endswith(f"/{name}")
    )
    return sorted(set(matches))


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = run_git(
        repo,
        "merge-base",
        "--is-ancestor",
        ancestor,
        descendant,
        check=False,
    )
    return result.returncode == 0


def commit_files(repo: Path, sha: str) -> list[str]:
    commit_and_parents = run_git(
        repo, "rev-list", "--parents", "-n", "1", sha
    ).stdout.split()
    if len(commit_and_parents) > 2:
        output = run_git(
            repo,
            "diff",
            "--name-only",
            "-M",
            commit_and_parents[1],
            sha,
        ).stdout
    else:
        output = run_git(
            repo,
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-M",
            sha,
        ).stdout
    return sorted({line for line in output.splitlines() if line})


def commit_record(repo: Path, sha: str, classification: str) -> dict[str, object]:
    subject = run_git(repo, "show", "-s", "--format=%s", sha).stdout.strip()
    files = commit_files(repo, sha)
    high_risk = sorted(
        path
        for path in files
        if any(marker in path for marker in HIGH_RISK_MARKERS)
    )
    return {
        "sha": sha,
        "subject": subject,
        "classification": classification,
        "files": files,
        "high_risk_paths": high_risk,
    }


def touches_jd_ci_boundary(files: Sequence[str]) -> bool:
    return any(
        path.startswith(prefix)
        for path in files
        for prefix in JD_CI_COMMIT_PATH_PREFIXES
    )


def atomic_write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RebaseError(f"cannot read JSON file {path}: {error}") from error


def build_plan(args: argparse.Namespace) -> dict[str, object]:
    repo = repository_root(Path(args.repo).resolve())
    old_internal = resolve_full_ref(repo, args.old_internal)
    old_upstream_name = args.old_upstream or infer_old_upstream(old_internal["ref"])
    old_upstream = resolve_full_ref(repo, old_upstream_name)
    new_upstream = resolve_full_ref(repo, args.new_upstream)
    new_internal_name = normalize_new_internal(
        args.new_internal or infer_new_internal(new_upstream["ref"])
    )

    existing = new_internal_exists(repo, new_internal_name)
    if existing:
        raise RebaseError(
            "new internal ref already exists: " + ", ".join(existing)
        )
    if not is_ancestor(repo, old_upstream["sha"], old_internal["sha"]):
        raise RebaseError(
            f"old upstream {old_upstream['ref']} is not an ancestor of "
            f"{old_internal['ref']}"
        )

    ordered = run_git(
        repo,
        "rev-list",
        "--reverse",
        "--topo-order",
        "--no-merges",
        f"{old_upstream['sha']}..{old_internal['sha']}",
    ).stdout.split()
    if not ordered:
        raise RebaseError("old JD delta is empty")
    merge_shas = run_git(
        repo,
        "rev-list",
        "--reverse",
        "--topo-order",
        "--merges",
        f"{old_upstream['sha']}..{old_internal['sha']}",
    ).stdout.split()
    audit_merges = [
        commit_record(repo, sha, "audit-merge") for sha in merge_shas
    ]

    cherry_output = run_git(
        repo,
        "cherry",
        new_upstream["sha"],
        old_internal["sha"],
        old_upstream["sha"],
    ).stdout
    signs = {}
    for line in cherry_output.splitlines():
        match = re.match(r"^([+-])\s+([0-9a-f]{40})$", line.strip())
        if match:
            signs[match.group(2)] = match.group(1)

    replay = []
    deferred_ci = []
    absorbed = []
    audit = []
    for sha in ordered:
        sign = signs.get(sha)
        if sign == "+":
            record = commit_record(repo, sha, "replay")
            if touches_jd_ci_boundary(record["files"]):
                record["classification"] = "deferred-ci"
                deferred_ci.append(record)
            else:
                replay.append(record)
        elif sign == "-":
            absorbed.append(commit_record(repo, sha, "absorbed"))
        elif is_ancestor(repo, sha, new_upstream["sha"]):
            absorbed.append(commit_record(repo, sha, "absorbed-exact"))
        else:
            audit.append(commit_record(repo, sha, "audit"))

    if audit:
        raise RebaseError(
            "patch classification missing for: "
            + ", ".join(item["sha"] for item in audit)
        )

    common_dir = git_common_dir(repo)
    worktree = (repo.parent / f"{repo.name}-{new_internal_name}").resolve()
    state_file = (common_dir / "jd-release-rebase" / f"{new_internal_name}.json").resolve()
    high_risk = sorted(
        {
            path
            for item in replay + deferred_ci + audit_merges
            for path in item["high_risk_paths"]
        }
    )
    return {
        "schema_version": 1,
        "repository": str(repo),
        "old_internal": old_internal,
        "old_upstream": old_upstream,
        "new_upstream": new_upstream,
        "new_internal": {
            "name": new_internal_name,
            "ref": f"refs/heads/{new_internal_name}",
        },
        "replay_commits": replay,
        "deferred_ci_commits": deferred_ci,
        "absorbed_commits": absorbed,
        "audit_merge_commits": audit_merges,
        "high_risk_paths": high_risk,
        "proposed_worktree": str(worktree),
        "proposed_state_file": str(state_file),
    }


def command_plan(args: argparse.Namespace) -> int:
    plan = build_plan(args)
    atomic_write_json(Path(args.output).resolve(), plan)
    return 0


def command_classify(args: argparse.Namespace) -> int:
    plan_file = Path(args.plan).resolve()
    plan = load_json(plan_file)
    old_sha = args.absorbed.strip()
    reason = args.reason.strip()
    if not SHA_PATTERN.fullmatch(old_sha):
        raise RebaseError(f"semantic absorbed commit must be a full SHA: {old_sha}")
    if not reason:
        raise RebaseError("semantic absorption requires a non-empty --reason")
    if Path(plan["proposed_state_file"]).exists():
        raise RebaseError(
            "release execution state already exists; classification is locked"
        )

    replay = plan["replay_commits"]
    matches = [item for item in replay if item["sha"] == old_sha]
    if len(matches) != 1:
        raise RebaseError(
            f"commit is not in the replay queue and cannot be reclassified: {old_sha}"
        )
    replay.remove(matches[0])
    absorbed = dict(matches[0])
    absorbed["classification"] = "absorbed-semantic"
    absorbed["absorption_reason"] = reason
    plan["absorbed_commits"].append(absorbed)
    atomic_write_json(plan_file, plan)
    sys.stdout.write(
        json.dumps(
            {
                "status": "updated",
                "plan": str(plan_file),
                "absorbed": old_sha,
                "reason": reason,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    return 0


def cherry_pick_in_progress(worktree: Path) -> bool:
    result = run_git(
        worktree, "rev-parse", "--quiet", "--verify", "CHERRY_PICK_HEAD", check=False
    )
    return result.returncode == 0


def unresolved_files(worktree: Path) -> list[str]:
    output = run_git(
        worktree, "diff", "--name-only", "--diff-filter=U", check=False
    ).stdout
    return sorted(line for line in output.splitlines() if line)


def state_path_from_args(
    plan: dict[str, object], state_file: str | None
) -> Path:
    return Path(state_file or plan["proposed_state_file"]).resolve()


def initialize_state(
    plan_file: Path,
    plan: dict[str, object],
    worktree: Path,
    state_file: Path,
) -> dict[str, object]:
    repo = Path(plan["repository"])
    existing = new_internal_exists(repo, plan["new_internal"]["name"])
    if existing:
        raise RebaseError(
            "new internal ref already exists: " + ", ".join(existing)
        )
    if worktree.exists():
        raise RebaseError(f"worktree path already exists: {worktree}")

    run_git(
        repo,
        "worktree",
        "add",
        "-b",
        plan["new_internal"]["name"],
        str(worktree),
        plan["new_upstream"]["sha"],
    )
    state = {
        "schema_version": 1,
        "status": "running",
        "repository": str(repo),
        "worktree": str(worktree),
        "plan_file": str(plan_file),
        "old_internal_sha": plan["old_internal"]["sha"],
        "old_upstream_sha": plan["old_upstream"]["sha"],
        "new_upstream_sha": plan["new_upstream"]["sha"],
        "new_internal": plan["new_internal"],
        "queue": plan["replay_commits"],
        "deferred_ci_commits": plan.get("deferred_ci_commits", []),
        "absorbed_commits": plan["absorbed_commits"],
        "audit_merge_commits": plan.get("audit_merge_commits", []),
        "high_risk_paths": plan["high_risk_paths"],
        "next_index": 0,
        "ci_next_index": 0,
        "ci_applied": [],
        "ci_skipped": [],
        "ci_pending_commit": None,
        "ci_commit_sha": None,
        "production_head": plan["new_upstream"]["sha"],
        "pending_commit": None,
        "mappings": {},
        "skipped": [],
        "conflict_files": [],
        "new_head": plan["new_upstream"]["sha"],
        "state_file": str(state_file),
    }
    atomic_write_json(state_file, state)
    return state


def persist_state(state: dict[str, object]) -> None:
    atomic_write_json(Path(state["state_file"]), state)


def record_success(state: dict[str, object], old_sha: str, new_sha: str) -> None:
    state["mappings"][old_sha] = new_sha
    state["new_head"] = new_sha
    state["next_index"] += 1
    state["pending_commit"] = None
    state["conflict_files"] = []
    state.pop("last_error", None)
    persist_state(state)


def record_skip(state: dict[str, object], old_sha: str, reason: str) -> None:
    state["skipped"].append({"sha": old_sha, "reason": reason})
    state["next_index"] += 1
    state["pending_commit"] = None
    state["conflict_files"] = []
    state.pop("last_error", None)
    persist_state(state)


def replay_remaining(state: dict[str, object]) -> int:
    worktree = Path(state["worktree"])
    queue = state["queue"]
    while state["next_index"] < len(queue):
        item = queue[state["next_index"]]
        old_sha = item["sha"]
        state["status"] = "running"
        state["pending_commit"] = old_sha
        state["conflict_files"] = []
        persist_state(state)

        result = run_git(worktree, "cherry-pick", old_sha, check=False)
        if result.returncode == 0:
            new_sha = run_git(worktree, "rev-parse", "HEAD").stdout.strip()
            record_success(state, old_sha, new_sha)
            continue

        conflicts = unresolved_files(worktree)
        if conflicts:
            state["status"] = "conflict"
            state["conflict_files"] = conflicts
            state["last_error"] = result.stderr.strip() or result.stdout.strip()
            persist_state(state)
            return CONFLICT_EXIT_CODE

        if cherry_pick_in_progress(worktree):
            cached = run_git(worktree, "diff", "--cached", "--quiet", check=False)
            if cached.returncode == 0:
                run_git(worktree, "cherry-pick", "--skip")
                record_skip(state, old_sha, "empty-during-replay")
                continue

        state["status"] = "error"
        state["last_error"] = result.stderr.strip() or result.stdout.strip()
        persist_state(state)
        raise RebaseError(
            f"cherry-pick failed without resolvable conflict for {old_sha}: "
            f"{state['last_error']}"
        )

    state["status"] = (
        "production-completed" if state["deferred_ci_commits"] else "completed"
    )
    state["pending_commit"] = None
    state["conflict_files"] = []
    state["new_head"] = run_git(worktree, "rev-parse", "HEAD").stdout.strip()
    state["production_head"] = state["new_head"]
    persist_state(state)
    return 0


def command_execute(args: argparse.Namespace) -> int:
    plan_file = Path(args.plan).resolve()
    plan = load_json(plan_file)
    repo = Path(plan["repository"])
    for field in ("old_internal", "old_upstream", "new_upstream"):
        current = resolve_full_ref(repo, plan[field]["ref"])
        if current["sha"] != plan[field]["sha"]:
            raise RebaseError(
                f"ref moved after planning: {plan[field]['ref']} "
                f"{plan[field]['sha']} -> {current['sha']}"
            )
    worktree = Path(args.worktree or plan["proposed_worktree"]).resolve()
    state_file = state_path_from_args(plan, args.state_file)
    state = initialize_state(plan_file, plan, worktree, state_file)
    return replay_remaining(state)


def command_resume(args: argparse.Namespace) -> int:
    state_file = Path(args.state_file).resolve()
    state = load_json(state_file)
    if state.get("status") != "conflict":
        raise RebaseError(
            f"state is not waiting on a conflict: {state.get('status')}"
        )
    worktree = Path(state["worktree"])
    conflicts = unresolved_files(worktree)
    if conflicts:
        raise RebaseError(
            "unresolved conflict files remain: " + ", ".join(conflicts)
        )
    old_sha = state.get("pending_commit")
    if not old_sha:
        raise RebaseError("conflict state has no pending commit")
    if not cherry_pick_in_progress(worktree):
        raise RebaseError(
            "CHERRY_PICK_HEAD is missing; do not continue manually before resume"
        )

    cached = run_git(worktree, "diff", "--cached", "--quiet", check=False)
    if cached.returncode == 0:
        run_git(worktree, "cherry-pick", "--skip")
        record_skip(state, old_sha, "resolved-to-empty")
    else:
        run_git(
            worktree,
            "-c",
            "core.editor=true",
            "cherry-pick",
            "--continue",
            extra_env={"GIT_EDITOR": "true"},
        )
        new_sha = run_git(worktree, "rev-parse", "HEAD").stdout.strip()
        record_success(state, old_sha, new_sha)
    return replay_remaining(state)


def prepare_deferred_ci(state: dict[str, object]) -> int:
    worktree = Path(state["worktree"])
    queue = state["deferred_ci_commits"]
    while state["ci_next_index"] < len(queue):
        item = queue[state["ci_next_index"]]
        old_sha = item["sha"]
        state["status"] = "ci-applying"
        state["ci_pending_commit"] = old_sha
        state["conflict_files"] = []
        persist_state(state)

        before_tree = run_git(worktree, "write-tree").stdout.strip()
        result = run_git(worktree, "cherry-pick", "--no-commit", old_sha, check=False)
        if result.returncode == 0:
            after_tree = run_git(worktree, "write-tree").stdout.strip()
            target = "ci_skipped" if before_tree == after_tree else "ci_applied"
            state[target].append(old_sha)
            state["ci_next_index"] += 1
            state["ci_pending_commit"] = None
            state["conflict_files"] = []
            state.pop("last_error", None)
            persist_state(state)
            continue

        conflicts = unresolved_files(worktree)
        if conflicts:
            state["status"] = "ci-conflict"
            state["conflict_files"] = conflicts
            state["last_error"] = result.stderr.strip() or result.stdout.strip()
            persist_state(state)
            return CONFLICT_EXIT_CODE

        state["status"] = "error"
        state["last_error"] = result.stderr.strip() or result.stdout.strip()
        persist_state(state)
        raise RebaseError(
            f"JD CI apply failed without resolvable conflict for {old_sha}: "
            f"{state['last_error']}"
        )

    state["status"] = "ci-prepared"
    state["ci_pending_commit"] = None
    state["conflict_files"] = []
    persist_state(state)
    return 0


def command_prepare_ci(args: argparse.Namespace) -> int:
    state = load_json(Path(args.state_file).resolve())
    if state.get("status") != "production-completed":
        raise RebaseError(
            "state is not ready for JD CI preparation: " + str(state.get("status"))
        )
    if not state.get("deferred_ci_commits"):
        raise RebaseError("release has no deferred JD CI commits")
    worktree = Path(state["worktree"])
    if run_git(worktree, "status", "--porcelain=v1").stdout.strip():
        raise RebaseError("worktree must be clean before JD CI preparation")
    return prepare_deferred_ci(state)


def command_resume_ci(args: argparse.Namespace) -> int:
    state = load_json(Path(args.state_file).resolve())
    if state.get("status") != "ci-conflict":
        raise RebaseError(
            "state is not waiting on a JD CI conflict: " + str(state.get("status"))
        )
    worktree = Path(state["worktree"])
    conflicts = unresolved_files(worktree)
    if conflicts:
        raise RebaseError(
            "unresolved JD CI conflict files remain: " + ", ".join(conflicts)
        )
    old_sha = state.get("ci_pending_commit")
    if not old_sha:
        raise RebaseError("JD CI conflict state has no pending commit")
    run_git(worktree, "cherry-pick", "--quit")
    state["ci_applied"].append(old_sha)
    state["ci_next_index"] += 1
    state["ci_pending_commit"] = None
    state["conflict_files"] = []
    state.pop("last_error", None)
    persist_state(state)
    return prepare_deferred_ci(state)


def command_commit_ci(args: argparse.Namespace) -> int:
    state = load_json(Path(args.state_file).resolve())
    if state.get("status") != "ci-prepared":
        raise RebaseError(
            "state is not ready for the final JD CI commit: "
            + str(state.get("status"))
        )
    worktree = Path(state["worktree"])
    head = run_git(worktree, "rev-parse", "HEAD").stdout.strip()
    if head != state["production_head"]:
        raise RebaseError("HEAD moved after production replay; JD CI must be the next commit")
    status = run_git(worktree, "status", "--porcelain=v1").stdout.splitlines()
    unstaged = [
        line
        for line in status
        if line.startswith("??") or (len(line) > 1 and line[1] != " ")
    ]
    if unstaged:
        raise RebaseError(
            "stage every intended JD CI change before commit-ci: " + ", ".join(unstaged)
        )
    cached = run_git(worktree, "diff", "--cached", "--quiet", check=False)
    if cached.returncode == 0:
        raise RebaseError("final JD CI commit has no staged changes")
    run_git(worktree, "commit", "-m", args.message)
    final_sha = run_git(worktree, "rev-parse", "HEAD").stdout.strip()
    for item in state["deferred_ci_commits"]:
        state["mappings"][item["sha"]] = final_sha
    state["ci_commit_sha"] = final_sha
    state["new_head"] = final_sha
    state["status"] = "completed"
    persist_state(state)
    return 0


def literal_strings(node: ast.AST) -> list[str]:
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if isinstance(item, str)]
    return []


def manifest_shas(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    shas = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = (
                node.targets[0]
                if isinstance(node, ast.Assign) and node.targets
                else node.target
            )
            value = node.value
            if isinstance(target, ast.Name) and target.id == "INTERNAL_COMMITS":
                shas.update(literal_strings(value))
        if isinstance(node, ast.Call):
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if name == "JDCase":
                for keyword in node.keywords:
                    if keyword.arg == "commits":
                        shas.update(literal_strings(keyword.value))
    return {sha for sha in shas if SHA_PATTERN.fullmatch(sha)}


def changed_and_untracked_files(worktree: Path, base: str, head: str) -> list[str]:
    committed = run_git(
        worktree, "diff", "--name-only", f"{base}..{head}"
    ).stdout.splitlines()
    status = run_git(worktree, "status", "--porcelain=v1").stdout.splitlines()
    working = []
    for line in status:
        path = line[3:] if len(line) > 3 else ""
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path:
            working.append(path)
    return sorted(set(committed + working))


def scan_conflict_markers(worktree: Path, paths: Sequence[str]) -> list[str]:
    findings = []
    for relative in paths:
        path = worktree / relative
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(lines, 1):
            if CONFLICT_MARKER.match(line):
                findings.append(f"{relative}:{number}:{line}")
    return findings


def command_check(args: argparse.Namespace) -> int:
    state = load_json(Path(args.state_file).resolve())
    worktree = Path(state["worktree"])
    head = run_git(worktree, "rev-parse", "HEAD").stdout.strip()
    ancestry_ok = is_ancestor(worktree, state["new_upstream_sha"], head)
    expected = {item["sha"] for item in state["queue"]}
    expected.update(item["sha"] for item in state.get("deferred_ci_commits", []))
    covered = set(state["mappings"])
    covered.update(item["sha"] for item in state["skipped"])
    mapping_complete = expected == covered

    range_diff_result = run_git(
        worktree,
        "range-diff",
        f"{state['old_upstream_sha']}..{state['old_internal_sha']}",
        f"{state['new_upstream_sha']}..{head}",
        check=False,
    )
    paths = changed_and_untracked_files(worktree, state["new_upstream_sha"], head)
    markers = scan_conflict_markers(worktree, paths)
    manifest = manifest_shas(worktree / "test/jd-ci/jd_test_manifest.py")
    old_shas = set(state["mappings"])
    old_shas.update(item["sha"] for item in state["absorbed_commits"])
    old_shas.update(item["sha"] for item in state["skipped"])
    production_old_shas = {item["sha"] for item in state["queue"]}
    new_shas = {
        new_sha
        for old_sha, new_sha in state["mappings"].items()
        if old_sha in production_old_shas
    }
    stale = sorted(old_shas & manifest)
    missing = sorted(new_shas - manifest)
    jd_ci_commits = run_git(
        worktree,
        "rev-list",
        "--reverse",
        f"{state['new_upstream_sha']}..{head}",
        "--",
        *JD_CI_COMMIT_PATH_PREFIXES,
    ).stdout.split()
    jd_ci_required = bool(state.get("deferred_ci_commits"))
    jd_ci_commit_is_head = (
        len(jd_ci_commits) == 1 and jd_ci_commits[0] == head
        if jd_ci_required
        else not jd_ci_commits
    )
    working_tree_clean = not run_git(
        worktree, "status", "--porcelain=v1"
    ).stdout.strip()

    passed = (
        state.get("status") == "completed"
        and ancestry_ok
        and mapping_complete
        and not markers
        and not stale
        and not missing
        and jd_ci_commit_is_head
        and (working_tree_clean or not jd_ci_required)
    )
    report = {
        "status": "passed" if passed else "failed",
        "state_status": state.get("status"),
        "new_head": head,
        "ancestry_ok": ancestry_ok,
        "mapping_complete": mapping_complete,
        "mapped_count": len(state["mappings"]),
        "skipped": state["skipped"],
        "range_diff_exit_code": range_diff_result.returncode,
        "range_diff": range_diff_result.stdout,
        "changed_files": paths,
        "conflict_markers": markers,
        "high_risk_paths": state["high_risk_paths"],
        "audit_merge_commits": state.get("audit_merge_commits", []),
        "manifest_stale_old_shas": stale,
        "manifest_missing_new_shas": missing,
        "jd_ci_commit_count": len(jd_ci_commits),
        "jd_ci_commits": jd_ci_commits,
        "jd_ci_commit_is_head": jd_ci_commit_is_head,
        "working_tree_clean": working_tree_clean,
    }
    if args.output:
        atomic_write_json(Path(args.output).resolve(), report)
    else:
        sys.stdout.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0 if passed else CHECK_FAILED_EXIT_CODE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, replay, consolidate JD CI, and audit a JD release rebase"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Create a read-only release replay plan")
    plan.add_argument("--repo", default=".")
    plan.add_argument("--old-internal", required=True)
    plan.add_argument("--old-upstream")
    plan.add_argument("--new-upstream", required=True)
    plan.add_argument("--new-internal")
    plan.add_argument("--output", required=True)
    plan.set_defaults(handler=command_plan)

    classify = subparsers.add_parser(
        "classify", help="Mark a replay patch as semantically absorbed upstream"
    )
    classify.add_argument("--plan", required=True)
    classify.add_argument("--absorbed", required=True)
    classify.add_argument("--reason", required=True)
    classify.set_defaults(handler=command_classify)

    execute = subparsers.add_parser("execute", help="Create a worktree and replay commits")
    execute.add_argument("--plan", required=True)
    execute.add_argument("--worktree")
    execute.add_argument("--state-file")
    execute.set_defaults(handler=command_execute)

    resume = subparsers.add_parser("resume", help="Continue after conflict resolution")
    resume.add_argument("--state-file", required=True)
    resume.set_defaults(handler=command_resume)

    prepare_ci = subparsers.add_parser(
        "prepare-ci", help="Apply all deferred JD CI commits without committing"
    )
    prepare_ci.add_argument("--state-file", required=True)
    prepare_ci.set_defaults(handler=command_prepare_ci)

    resume_ci = subparsers.add_parser(
        "resume-ci", help="Continue deferred JD CI application after conflict resolution"
    )
    resume_ci.add_argument("--state-file", required=True)
    resume_ci.set_defaults(handler=command_resume_ci)

    commit_ci = subparsers.add_parser(
        "commit-ci", help="Create the single final JD CI commit at HEAD"
    )
    commit_ci.add_argument("--state-file", required=True)
    commit_ci.add_argument("--message", required=True)
    commit_ci.set_defaults(handler=command_commit_ci)

    check = subparsers.add_parser("check", help="Audit a completed release replay")
    check.add_argument("--state-file", required=True)
    check.add_argument("--output")
    check.set_defaults(handler=command_check)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (RebaseError, OSError, SyntaxError) as error:
        print(f"jd_release_rebase: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
