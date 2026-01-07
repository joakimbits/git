#!/usr/bin/env python3
# file: git-tree.py

from __future__ import annotations

import argparse
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ------------------------- model -------------------------


@dataclass(frozen=True)
class CommitMeta:
    sha: str
    author: str
    date_iso: str
    subject: str
    epoch: int


@dataclass(frozen=True)
class HistoryGraph:
    shas: List[str]
    parents: Dict[str, List[str]]  # compressed to nearest-in-set parents
    metas: Dict[str, CommitMeta]
    branch_labels: Dict[str, List[str]]  # sha -> [refname...]
    tag_labels: Dict[str, List[str]]     # sha -> [tag...]
    head_tip: str
    head_branch: str


# ------------------------- git runner -------------------------


class Git:
    """Minimal git runner; crashes on git failure and lets git print stderr."""

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_s: int = 60,
        stdin_text: Optional[str] = None,
    ) -> str:
        cp = subprocess.run(
            ["git", "-C", str(self.cwd), *args],
            input=stdin_text,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            check=True,
            timeout=timeout_s,
        )
        return cp.stdout

    def run_passthru(self, args: Sequence[str], *, timeout_s: int = 0) -> None:
        subprocess.run(
            ["git", "-C", str(self.cwd), *args],
            stdout=None,
            stderr=None,
            text=True,
            check=True,
            timeout=timeout_s if timeout_s > 0 else None,
        )

    def toplevel(self) -> Path:
        return Path(self.run(["rev-parse", "--show-toplevel"], timeout_s=30).strip())


# ------------------------- argument passthrough split -------------------------


_DIFF_FLAG_SET = {
    "--name-only",
    "--name-status",
    "--stat",
    "--numstat",
    "--shortstat",
    "--patch",
    "-p",
    "-w",
    "-b",
    "--ignore-space-change",
    "--ignore-all-space",
    "--ignore-space-at-eol",
    "--ignore-cr-at-eol",
    "--minimal",
    "--patience",
    "--histogram",
    "--find-copies-harder",
    "--no-renames",
    "-M",
    "-C",
}

_DIFF_PREFIXES = (
    "--diff-filter=",
    "--find-renames",
    "--find-renames=",
    "--find-copies",
    "--find-copies=",
    "--diff-algorithm=",
    "--word-diff",
    "--word-diff=",
    "--unified=",
    "--color",
    "--color=",
)

_ORDER_FLAGS = {
    "--topo-order",
    "--date-order",
    "--author-date-order",
}


def _split_git_and_pathspec(rest: List[str]) -> Tuple[List[str], List[str]]:
    if "--" in rest:
        i = rest.index("--")
        return rest[:i], [p for p in rest[i + 1 :] if p.strip()]
    return rest, []


def _is_diffish(arg: str) -> bool:
    if arg in _DIFF_FLAG_SET:
        return True
    if any(arg.startswith(p) for p in _DIFF_PREFIXES):
        return True
    if arg.startswith("-U") and len(arg) > 2:  # e.g. -U3
        return True
    return False


def _extract_max_count(args: List[str], default_n: int) -> int:
    n = default_n
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-n" and i + 1 < len(args):
            try:
                n = int(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        if a.startswith("-n") and len(a) > 2:
            try:
                n = int(a[2:])
            except ValueError:
                pass
            i += 1
            continue
        if a == "--max-count" and i + 1 < len(args):
            try:
                n = int(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        if a.startswith("--max-count="):
            try:
                n = int(a.split("=", 1)[1])
            except ValueError:
                pass
            i += 1
            continue
        if a.startswith("-") and len(a) > 1 and a[1:].isdigit():  # git shorthand -20
            try:
                n = int(a[1:])
            except ValueError:
                pass
            i += 1
            continue
        i += 1
    return max(1, n)


def _has_order_flag(args: List[str]) -> bool:
    return any(a in _ORDER_FLAGS for a in args)


def _pick_output_mode(git_args: List[str]) -> Tuple[bool, bool]:
    # Follow "last one wins" behavior if user passed both.
    mode = None
    for a in git_args:
        if a == "--name-only":
            mode = "name-only"
        elif a == "--name-status":
            mode = "name-status"
    return (mode == "name-only"), (mode == "name-status")


def _partition_args_for_tree_mode(git_args: List[str]) -> Tuple[List[str], List[str]]:
    """
    Returns:
      log_args: raw passthrough args suitable for `git log` commit-listing (no diff output)
      diff_args: raw passthrough args suitable for `git diff`/`git diff-tree`
    """
    log_args: List[str] = []
    diff_args: List[str] = []

    for a in git_args:
        if _is_diffish(a):
            diff_args.append(a)
            continue
        log_args.append(a)

    # Also remove name-only/name-status from diff_args; we control those per command.
    diff_args = [a for a in diff_args if a not in ("--name-only", "--name-status")]
    return log_args, diff_args


# ------------------------- graph builder -------------------------


class GraphBuilder:
    def __init__(self, git: Git) -> None:
        self.git = git

    def build(
        self,
        *,
        git_log_args: List[str],
        pathspecs: List[str],
        max_commits: int,
        include_remotes_in_labels: bool = True,
    ) -> HistoryGraph:
        head_branch = self.git.run(["rev-parse", "--abbrev-ref", "HEAD"], timeout_s=30).strip()
        head_tip = self._head_tip(pathspecs)

        rows = self._log_rows(
            git_log_args=git_log_args,
            pathspecs=pathspecs,
            max_commits=max_commits,
        )
        shas = [sha for sha, _ in rows]
        sha_set = set(shas)

        parents_map = self._ancestor_parent_graph(shas)

        nearest_cache: Dict[str, List[str]] = {}
        parents: Dict[str, List[str]] = {}
        for sha, direct_parents in rows:
            out: List[str] = []
            for p in direct_parents:
                out.extend(
                    self._nearest_in_set(
                        p,
                        sha_set=sha_set,
                        parents_map=parents_map,
                        cache=nearest_cache,
                    )
                )
            uniq: List[str] = []
            seen: Set[str] = set()
            for x in out:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            parents[sha] = uniq

        metas = self._fetch_meta(shas)

        branch_labels = self._branch_labels(
            pathspecs,
            include_locals=True,
            include_remotes=include_remotes_in_labels,
        )
        tag_labels = self._tag_labels(pathspecs)

        return HistoryGraph(
            shas=shas,
            parents=parents,
            metas=metas,
            branch_labels=branch_labels,
            tag_labels=tag_labels,
            head_tip=head_tip,
            head_branch=head_branch,
        )

    def _head_tip(self, pathspecs: List[str]) -> str:
        if not pathspecs:
            return self.git.run(["rev-parse", "HEAD"], timeout_s=30).strip()
        out = self.git.run(["log", "-n", "1", "--pretty=format:%H", "HEAD", "--", *pathspecs], timeout_s=120).strip()
        return out.splitlines()[0].strip() if out else ""

    def _log_rows(
        self,
        *,
        git_log_args: List[str],
        pathspecs: List[str],
        max_commits: int,
    ) -> List[Tuple[str, List[str]]]:
        # Force parseable output. Ensure no diff output is printed.
        cmd = [
            "log",
            *git_log_args,
        ]
        if not _has_order_flag(git_log_args):
            cmd.append("--topo-order")
        cmd += [
            f"-n{max_commits}",
            "--no-patch",
            "--pretty=format:%H %P",
        ]
        if pathspecs:
            cmd += ["--", *pathspecs]

        out = self.git.run(cmd, timeout_s=240)
        rows: List[Tuple[str, List[str]]] = []
        for ln in out.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            toks = ln.split()
            rows.append((toks[0], toks[1:]))
        return rows

    def _ancestor_parent_graph(self, start_commits: List[str]) -> Dict[str, List[str]]:
        out = self.git.run(
            ["rev-list", "--parents", "--topo-order", "--stdin"],
            timeout_s=240,
            stdin_text="\n".join(start_commits) + "\n",
        )
        parents_map: Dict[str, List[str]] = {}
        for ln in out.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            toks = ln.split()
            parents_map[toks[0]] = toks[1:]
        return parents_map

    def _nearest_in_set(
        self,
        start_sha: str,
        *,
        sha_set: Set[str],
        parents_map: Dict[str, List[str]],
        cache: Dict[str, List[str]],
        max_bfs: int = 50_000,
    ) -> List[str]:
        if not start_sha:
            return []
        if start_sha in cache:
            return cache[start_sha]
        if start_sha in sha_set:
            cache[start_sha] = [start_sha]
            return cache[start_sha]

        q: Deque[Tuple[str, int]] = deque([(start_sha, 0)])
        visited: Set[str] = set()
        best_depth: Optional[int] = None
        found: List[str] = []
        steps = 0

        while q:
            sha, depth = q.popleft()
            steps += 1
            if steps > max_bfs:
                break
            if sha in visited:
                continue
            visited.add(sha)

            if best_depth is not None and depth > best_depth:
                break

            if sha in sha_set:
                best_depth = depth if best_depth is None else best_depth
                if depth == best_depth:
                    found.append(sha)
                continue

            for p in parents_map.get(sha, []):
                if p and p not in visited:
                    q.append((p, depth + 1))

        uniq: List[str] = []
        seen: Set[str] = set()
        for x in found:
            if x not in seen:
                seen.add(x)
                uniq.append(x)

        cache[start_sha] = uniq
        return uniq

    def _fetch_meta(self, shas: List[str]) -> Dict[str, CommitMeta]:
        if not shas:
            return {}
        fmt = "%H%x1f%an%x1f%ad%x1f%ct%x1f%s%x1e"
        metas: Dict[str, CommitMeta] = {}
        for chunk in self._chunked(list(dict.fromkeys(shas)), 200):
            out = self.git.run(
                ["show", "-s", "--date=iso-strict", f"--pretty=format:{fmt}", *chunk],
                timeout_s=240,
            )
            for rec in out.split("\x1e"):
                rec = rec.strip()
                if not rec:
                    continue
                parts = rec.split("\x1f")
                if len(parts) != 5:
                    continue
                sha, author, date_iso, epoch_s, subject = (p.strip() for p in parts)
                metas[sha] = CommitMeta(
                    sha=sha,
                    author=author,
                    date_iso=date_iso,
                    subject=subject,
                    epoch=int(epoch_s),
                )
        return metas

    def _refs_list(self, *, include_locals: bool, include_remotes: bool) -> List[str]:
        refs: List[str] = []
        if include_locals:
            refs.append("refs/heads")
        if include_remotes:
            refs.append("refs/remotes")
        if not refs:
            return []
        out = self.git.run(["for-each-ref", "--format=%(refname:short)", *refs], timeout_s=60)
        names = [ln.strip() for ln in out.splitlines() if ln.strip()]
        names = [n for n in names if not n.endswith("/HEAD")]
        names.sort(key=str.casefold)
        return names

    def _tip_sha(self, ref: str, pathspecs: List[str]) -> str:
        if not pathspecs:
            return self.git.run(["rev-parse", ref], timeout_s=60).strip()
        out = self.git.run(["log", "-n", "1", "--pretty=format:%H", ref, "--", *pathspecs], timeout_s=120).strip()
        return out.splitlines()[0].strip() if out else ""

    def _branch_labels(
        self,
        pathspecs: List[str],
        *,
        include_locals: bool,
        include_remotes: bool,
    ) -> Dict[str, List[str]]:
        commit_to_branches: Dict[str, List[str]] = {}
        for b in self._refs_list(include_locals=include_locals, include_remotes=include_remotes):
            tip = self._tip_sha(b, pathspecs)
            if tip:
                commit_to_branches.setdefault(tip, []).append(b)
        for sha in commit_to_branches:
            commit_to_branches[sha].sort(key=str.casefold)
        return commit_to_branches

    def _tag_labels(self, pathspecs: List[str]) -> Dict[str, List[str]]:
        tags_out = self.git.run(["tag", "--list"], timeout_s=120)
        tags = [t.strip() for t in tags_out.splitlines() if t.strip()]
        commit_to_tags: Dict[str, List[str]] = {}
        for tag in tags:
            tip = self._tip_sha(tag, pathspecs)
            if tip:
                commit_to_tags.setdefault(tip, []).append(tag)
        for sha in commit_to_tags:
            commit_to_tags[sha].sort(key=str.casefold)
        return commit_to_tags

    @staticmethod
    def _chunked(xs: List[str], n: int) -> Iterable[List[str]]:
        for i in range(0, len(xs), n):
            yield xs[i : i + n]


# ------------------------- touched files provider (diff passthrough) -------------------------


class EdgeTouchedFiles:
    """
    Compute touched files for a node by comparing it with its *visible parent*.

    Root node:
      git diff-tree --root -r --name-status <sha> <diff_args...> -- <pathspec...>

    Edge parent->node:
      git diff --name-status <parent> <sha> <diff_args...> -- <pathspec...>
    """

    def __init__(self, git: Git, pathspecs: List[str], diff_args: List[str]) -> None:
        self.git = git
        self.pathspecs = pathspecs
        self.diff_args = diff_args
        self._cache: Dict[Tuple[Optional[str], str, bool, Tuple[str, ...]], List[str]] = {}

    def get(self, sha: str, *, parent_sha: Optional[str], name_status: bool) -> List[str]:
        key = (parent_sha, sha, name_status, tuple(self.diff_args))
        if key in self._cache:
            return self._cache[key]

        if parent_sha:
            cmd = [
                "diff",
                *self.diff_args,
                "--name-status" if name_status else "--name-only",
                parent_sha,
                sha,
            ]
        else:
            cmd = [
                "diff-tree",
                "--root",
                "-r",
                "--no-commit-id",
                *self.diff_args,
                "--name-status" if name_status else "--name-only",
                sha,
            ]

        if self.pathspecs:
            cmd += ["--", *self.pathspecs]

        out = self.git.run(cmd, timeout_s=240)
        lines = [ln.rstrip("\n") for ln in out.splitlines() if ln.strip()]
        self._cache[key] = lines
        return lines


# ------------------------- printing -------------------------


@dataclass(frozen=True)
class TreeStyle:
    tee: str
    elbow: str
    vert: str
    space: str

    @staticmethod
    def unicode() -> "TreeStyle":
        return TreeStyle(tee="├─ ", elbow="└─ ", vert="│   ", space="    ")

    @staticmethod
    def ascii() -> "TreeStyle":
        return TreeStyle(tee="|-- ", elbow="`-- ", vert="|   ", space="    ")


class TreePrinter:
    def __init__(
        self,
        g: HistoryGraph,
        *,
        style: TreeStyle,
        compact: bool,
        supercompact: bool,
        touched: Optional[EdgeTouchedFiles],
        show_name_only: bool,
        show_name_status: bool,
        max_files: int,
    ) -> None:
        self.g = g
        self.style = style
        self.compact = bool(compact or supercompact)
        self.supercompact = bool(supercompact)
        self.touched = touched
        self.show_name_only = show_name_only
        self.show_name_status = show_name_status
        self.max_files = max(0, int(max_files))

    def print(self) -> None:
        g = self.g
        if not g.shas:
            print("No commits in selection.")
            return

        children_of: Dict[str, List[str]] = {sha: [] for sha in g.shas}
        roots: List[str] = []

        for child in g.shas:
            ps = g.parents.get(child, [])
            if not ps:
                roots.append(child)
            for p in ps:
                if p in children_of:
                    children_of[p].append(child)

        epoch = {sha: g.metas.get(sha).epoch if sha in g.metas else 0 for sha in g.shas}
        for p, kids in children_of.items():
            kids.sort(key=lambda s: epoch.get(s, 0), reverse=True)

        roots_sorted = sorted(roots, key=lambda s: epoch.get(s, 0))
        primary_root = roots_sorted[0]

        if g.head_tip and g.head_tip in set(g.shas):
            cur = g.head_tip
            seen: Set[str] = set()
            while True:
                if cur in seen:
                    break
                seen.add(cur)
                ps = g.parents.get(cur, [])
                if not ps:
                    primary_root = cur
                    break
                cur = min(ps, key=lambda x: epoch.get(x, 0))

        roots_ordered = [primary_root] + [r for r in roots_sorted if r != primary_root]
        self._walk_forest(roots_ordered, children_of)

    def _is_key_decorated(self, sha: str) -> bool:
        g = self.g
        if sha == g.head_tip:
            return True
        if sha in g.branch_labels:
            return True
        if not self.supercompact and sha in g.tag_labels:
            return True
        return False

    def _is_key_node(self, sha: str, children_of: Dict[str, List[str]], roots: Set[str]) -> bool:
        if sha in roots:
            return True
        if self._is_key_decorated(sha):
            return True
        return len(children_of.get(sha, [])) >= 2

    def _fmt_commit(self, sha: str) -> str:
        g = self.g
        m = g.metas.get(sha)
        short = sha[:10]
        date = (m.date_iso[:10] if m and m.date_iso else "")
        subj = (m.subject if m else "")

        pre: List[str] = []
        if sha == g.head_tip:
            pre.append("HEAD")
        bs = ", ".join(g.branch_labels.get(sha, []))
        if bs:
            pre.append(bs)
        ts = ", ".join(g.tag_labels.get(sha, []))
        if ts:
            pre.append(f"tags:{ts}")

        prefix = f"[{' | '.join(pre)}] " if pre else ""
        return f"{prefix}{short}  {date}  {subj}".rstrip()

    def _walk_forest(self, roots: List[str], children_of: Dict[str, List[str]]) -> None:
        show_files = (self.show_name_only or self.show_name_status) and (self.touched is not None)
        name_status = self.show_name_status
        roots_set = set(roots)

        def compress_chain(start: str) -> str:
            if not self.compact:
                return start
            cur = start
            while True:
                if self._is_key_node(cur, children_of, roots=roots_set):
                    return cur
                kids = children_of.get(cur, [])
                if len(kids) != 1:
                    return cur
                cur = kids[0]

        def print_files(prefix: str, sha: str, parent_sha: Optional[str]) -> None:
            if not show_files:
                return
            lines = self.touched.get(sha, parent_sha=parent_sha, name_status=name_status)

            if self.max_files and len(lines) > self.max_files:
                shown = lines[: self.max_files]
                more = len(lines) - self.max_files
            else:
                shown = lines
                more = 0

            for ln in shown:
                print(f"{prefix}· {ln}")
            if more:
                print(f"{prefix}· ... ({more} more)")

        def walk(node: str, *, prefix: str, is_last: bool, path: Set[str], visible_parent: Optional[str]) -> None:
            conn = self.style.elbow if is_last else self.style.tee
            line = self._fmt_commit(node)
            if prefix:
                print(f"{prefix}{conn}{line}")
            else:
                print(line)

            if node in path:
                return
            path2 = set(path)
            path2.add(node)

            child_prefix = prefix + (self.style.space if is_last else self.style.vert)
            print_files(child_prefix, node, visible_parent)

            kids = children_of.get(node, [])
            if not kids:
                return

            ends: List[str] = []
            seen_end: Set[str] = set()
            for k in kids:
                end = compress_chain(k)
                if end not in seen_end:
                    seen_end.add(end)
                    ends.append(end)

            for i, end in enumerate(ends):
                walk(
                    end,
                    prefix=child_prefix,
                    is_last=(i == len(ends) - 1),
                    path=path2,
                    visible_parent=node,
                )

        for i, r in enumerate(roots):
            walk(r, prefix="", is_last=(i == len(roots) - 1), path=set(), visible_parent=None)


# ------------------------- CLI -------------------------


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("-C", dest="C", default=".", help="run as if started in <path>")
    ap.add_argument("--graph", default="tree", choices=["tree", "dag"])
    ap.add_argument("--graph-style", default="unicode", choices=["unicode", "ascii"])
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--supercompact", action="store_true")
    ap.add_argument("--max-files", type=int, default=0, help="0=unlimited")
    ns, rest = ap.parse_known_args()

    git_args, pathspecs = _split_git_and_pathspec(rest)
    pathspecs = [p.replace("\\", "/") for p in pathspecs]

    show_name_only, show_name_status = _pick_output_mode(git_args)

    cwd = Path(ns.C).expanduser().resolve()
    git = Git(cwd)
    repo = git.toplevel()

    if ns.graph == "dag":
        # Pure passthrough to git log (your args define everything).
        cmd = ["log", *git_args]
        if pathspecs:
            cmd += ["--", *pathspecs]
        print(f"Repo: {repo}")
        print(f"Args: {' '.join(git_args) if git_args else '(none)'}")
        print(f"Pathspec: {pathspecs if pathspecs else '(none)'}")
        print()
        git.run_passthru(cmd)
        return

    # Tree mode:
    log_args, diff_args = _partition_args_for_tree_mode(git_args)

    # If user didn't request touch output, don't run diffs.
    touched = None
    if show_name_only or show_name_status:
        touched = EdgeTouchedFiles(git, pathspecs, diff_args=diff_args)

    max_commits = _extract_max_count(git_args, default_n=2000)

    g = GraphBuilder(git).build(
        git_log_args=log_args,
        pathspecs=pathspecs,
        max_commits=max_commits,
        include_remotes_in_labels=True,
    )

    style = TreeStyle.unicode() if ns.graph_style == "unicode" else TreeStyle.ascii()

    print(f"Repo: {repo}")
    print(f"Args: {' '.join(git_args) if git_args else '(none)'}")
    print(f"Pathspec: {pathspecs if pathspecs else '(none)'}")
    print(f"HEAD branch: {g.head_branch}")
    print(f"Commits: {len(g.shas)}")
    if ns.supercompact:
        print("Mode: supercompact (tags do not affect tree shape)")
    elif ns.compact:
        print("Mode: compact")
    if diff_args:
        print(f"Diff args (passthrough): {' '.join(diff_args)}")
    print()

    TreePrinter(
        g,
        style=style,
        compact=ns.compact,
        supercompact=ns.supercompact,
        touched=touched,
        show_name_only=show_name_only,
        show_name_status=show_name_status,
        max_files=ns.max_files,
    ).print()


if __name__ == "__main__":
    main()
