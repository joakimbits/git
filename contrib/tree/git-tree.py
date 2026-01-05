#!/usr/bin/env python3
# file: git-tree.py

"""
Print the basic file-history tree (touch-commit graph) to stdout.

Git-like CLI:
  python git-tree.py [-C <path>] [--max-commits N]
                     [--all] [--branches[=<pat>]] [--remotes[=<pat>]] [--tags[=<pat>]]
                     [--name-only | --name-status] [--max-files N]
                     -- <pathspec...>

Semantics:
- If no ref selector is provided, the revision set is the default git log set (HEAD).
- If any selector is provided, the revision set is the union of those selectors.
- Pathspecs are passed to git after `--` exactly as provided.
- Crashy by design: git errors print to stderr; Python raises on failures.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_SENTINEL_ALL = "__ALL__"


# ------------------------- model -------------------------


@dataclass(frozen=True)
class CommitMeta:
    sha: str
    author: str
    date_iso: str
    subject: str
    epoch: int


@dataclass(frozen=True)
class FileHistoryGraph:
    touch_shas: List[str]
    touch_parents: Dict[str, List[str]]
    metas: Dict[str, CommitMeta]
    branch_labels: Dict[str, List[str]]
    tag_labels: Dict[str, List[str]]
    head_file_tip: str
    head_branch: str


@dataclass(frozen=True)
class RefSelectors:
    all: bool
    branches: List[str]  # entries are _SENTINEL_ALL or patterns
    remotes: List[str]   # entries are _SENTINEL_ALL or patterns
    tags: List[str]      # entries are _SENTINEL_ALL or patterns

    def any(self) -> bool:
        return self.all or bool(self.branches) or bool(self.remotes) or bool(self.tags)

    def to_git_args(self) -> List[str]:
        args: List[str] = []
        if self.all:
            args.append("--all")
        for p in self.branches:
            args.append("--branches" if p == _SENTINEL_ALL else f"--branches={p}")
        for p in self.remotes:
            args.append("--remotes" if p == _SENTINEL_ALL else f"--remotes={p}")
        for p in self.tags:
            args.append("--tags" if p == _SENTINEL_ALL else f"--tags={p}")
        return args


# ------------------------- git runner -------------------------


class Git:
    """Minimal git runner that crashes on errors and inherits stderr."""

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
            stderr=None,  # let git print errors
            text=True,
            check=True,
            timeout=timeout_s,
        )
        return cp.stdout

    def show_toplevel(self) -> Path:
        out = self.run(["rev-parse", "--show-toplevel"], timeout_s=30).strip()
        return Path(out)


# ------------------------- builder -------------------------


class FileHistoryGraphBuilder:
    """Builds the touch-commit graph and labels for a given pathspec list."""

    def __init__(self, git: Git) -> None:
        self.git = git

    def build(
        self,
        pathspecs: List[str],
        *,
        max_commits: int = 2000,
        selectors: RefSelectors,
    ) -> FileHistoryGraph:
        head_branch = self._head_branch()
        head_file_tip = self._head_file_tip_sha(pathspecs)

        touch_rows = self._touch_log_rows(pathspecs, max_commits=max_commits, selectors=selectors)
        touch_shas = [sha for sha, _ in touch_rows]
        touch_set = set(touch_shas)

        parents_map = self._ancestor_parent_graph(touch_shas)

        nearest_cache: Dict[str, List[str]] = {}
        touch_parents: Dict[str, List[str]] = {}
        for sha, direct_parents in touch_rows:
            parents_out: List[str] = []
            for p in direct_parents:
                parents_out.extend(
                    self._nearest_touch_commits(
                        p,
                        touch_set=touch_set,
                        parents_map=parents_map,
                        cache=nearest_cache,
                    )
                )
            uniq: List[str] = []
            seen: Set[str] = set()
            for x in parents_out:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            touch_parents[sha] = uniq

        metas = self._fetch_meta(touch_shas)

        # Labels: if selectors were explicitly provided, match the categories.
        # If none were provided (HEAD-only), still label local branches + tags (useful context).
        label_branches = selectors.all or bool(selectors.branches) or (not selectors.any())
        label_remotes = selectors.all or bool(selectors.remotes)
        label_tags = selectors.all or bool(selectors.tags) or (not selectors.any())

        branch_labels = self._branch_labels_for_file(
            pathspecs,
            include_locals=label_branches,
            include_remotes=label_remotes,
        )
        tag_labels = self._tag_labels_for_file(pathspecs) if label_tags else {}

        return FileHistoryGraph(
            touch_shas=touch_shas,
            touch_parents=touch_parents,
            metas=metas,
            branch_labels=branch_labels,
            tag_labels=tag_labels,
            head_file_tip=head_file_tip,
            head_branch=head_branch,
        )

    def _head_branch(self) -> str:
        return self.git.run(["rev-parse", "--abbrev-ref", "HEAD"], timeout_s=30).strip()

    def _head_file_tip_sha(self, pathspecs: List[str]) -> str:
        out = self.git.run(
            ["log", "-n", "1", "--pretty=format:%H", "HEAD", "--", *pathspecs],
            timeout_s=120,
        ).strip()
        return out.splitlines()[0].strip() if out else ""

    def _touch_log_rows(
        self,
        pathspecs: List[str],
        *,
        max_commits: int,
        selectors: RefSelectors,
    ) -> List[Tuple[str, List[str]]]:
        # Ref selectors are standard git options: --all/--branches/--remotes/--tags.
        # If none provided, git log defaults to HEAD.
        cmd = ["log", *selectors.to_git_args(), "--topo-order", f"-n{max_commits}", "--pretty=format:%H %P", "--", *pathspecs]
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

    def _nearest_touch_commits(
        self,
        start_sha: str,
        *,
        touch_set: Set[str],
        parents_map: Dict[str, List[str]],
        cache: Dict[str, List[str]],
        max_bfs: int = 50_000,
    ) -> List[str]:
        if not start_sha:
            return []
        if start_sha in cache:
            return cache[start_sha]
        if start_sha in touch_set:
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

            if sha in touch_set:
                if best_depth is None:
                    best_depth = depth
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

    def _branches_list(self, *, include_locals: bool, include_remotes: bool) -> List[str]:
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

    def _file_tip_sha(self, ref: str, pathspecs: List[str]) -> str:
        out = self.git.run(
            ["log", "-n", "1", "--pretty=format:%H", ref, "--", *pathspecs],
            timeout_s=120,
        ).strip()
        return out.splitlines()[0].strip() if out else ""

    def _branch_labels_for_file(
        self,
        pathspecs: List[str],
        *,
        include_locals: bool,
        include_remotes: bool,
    ) -> Dict[str, List[str]]:
        commit_to_branches: Dict[str, List[str]] = {}
        for b in self._branches_list(include_locals=include_locals, include_remotes=include_remotes):
            tip = self._file_tip_sha(b, pathspecs)
            if tip:
                commit_to_branches.setdefault(tip, []).append(b)
        for sha in commit_to_branches:
            commit_to_branches[sha].sort(key=str.casefold)
        return commit_to_branches

    def _tag_labels_for_file(self, pathspecs: List[str]) -> Dict[str, List[str]]:
        tags_out = self.git.run(["tag", "--list"], timeout_s=120)
        tags = [t.strip() for t in tags_out.splitlines() if t.strip()]

        commit_to_tags: Dict[str, List[str]] = {}
        for tag in tags:
            sha = self.git.run(
                ["log", "-n", "1", "--pretty=format:%H", tag, "--", *pathspecs],
                timeout_s=120,
            ).strip()
            if sha:
                commit_to_tags.setdefault(sha, []).append(tag)

        for sha in commit_to_tags:
            commit_to_tags[sha].sort(key=str.casefold)
        return commit_to_tags

    @staticmethod
    def _chunked(xs: List[str], n: int) -> Iterable[List[str]]:
        for i in range(0, len(xs), n):
            yield xs[i : i + n]


# ------------------------- touched files provider -------------------------


class CommitTouchedFiles:
    """Per-commit touched-paths provider (cached)."""

    def __init__(self, git: Git, pathspecs: List[str]) -> None:
        self.git = git
        self.pathspecs = pathspecs
        self._cache_name_only: Dict[str, List[str]] = {}
        self._cache_name_status: Dict[str, List[str]] = {}

    def get(self, sha: str, *, name_status: bool) -> List[str]:
        cache = self._cache_name_status if name_status else self._cache_name_only
        if sha in cache:
            return cache[sha]

        args = [
            "diff-tree",
            "--root",
            "-m",
            "-r",
            "--no-commit-id",
            "--name-status" if name_status else "--name-only",
            sha,
            "--",
            *self.pathspecs,
        ]
        out = self.git.run(args, timeout_s=120)

        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        seen: Set[str] = set()
        uniq: List[str] = []
        for ln in lines:
            if ln not in seen:
                seen.add(ln)
                uniq.append(ln)

        cache[sha] = uniq
        return uniq


# ------------------------- printer -------------------------


class AsciiTreePrinter:
    """Print a rooted forest of commits using the compressed parent graph."""

    def __init__(
        self,
        graph: FileHistoryGraph,
        *,
        touched_files: Optional[CommitTouchedFiles] = None,
        show_name_only: bool = False,
        show_name_status: bool = False,
        max_files: int = 0,
    ) -> None:
        self.g = graph
        self.touched_files = touched_files
        self.show_name_only = show_name_only
        self.show_name_status = show_name_status
        self.max_files = max(0, int(max_files))

    def print(self) -> None:
        g = self.g
        if not g.touch_shas:
            print("No commits touch that pathspec.")
            return

        children_of: Dict[str, List[str]] = {sha: [] for sha in g.touch_shas}
        roots: List[str] = []

        for child in g.touch_shas:
            parents = g.touch_parents.get(child, [])
            if not parents:
                roots.append(child)
            for p in parents:
                if p in children_of:
                    children_of[p].append(child)

        epoch_map = {sha: g.metas.get(sha).epoch if sha in g.metas else 0 for sha in g.touch_shas}
        for p, kids in children_of.items():
            kids.sort(key=lambda s: epoch_map.get(s, 0), reverse=True)

        root_order = sorted(roots, key=lambda s: epoch_map.get(s, 0))
        primary_root = root_order[0]

        if g.head_file_tip and g.head_file_tip in set(g.touch_shas):
            cur = g.head_file_tip
            seen: Set[str] = set()
            while True:
                if cur in seen:
                    break
                seen.add(cur)
                ps = g.touch_parents.get(cur, [])
                if not ps:
                    primary_root = cur
                    break
                cur = min(ps, key=lambda x: epoch_map.get(x, 0))

        ordered_roots = [primary_root] + [r for r in root_order if r != primary_root]

        if len(ordered_roots) == 1:
            self._print_tree(ordered_roots, children_of)
            return

        self._print_tree([primary_root], children_of)
        print()
        print(f"Other roots ({len(ordered_roots) - 1})")
        self._print_tree(ordered_roots[1:], children_of)

    def _format_commit_line(self, sha: str) -> str:
        g = self.g
        m = g.metas.get(sha)
        short = sha[:10]
        date_part = (m.date_iso[:10] if m and m.date_iso else "")
        subject = (m.subject if m else "")

        pre: List[str] = []
        if sha == g.head_file_tip:
            pre.append("HEAD")
        b = ", ".join(g.branch_labels.get(sha, []))
        if b:
            pre.append(b)
        t = ", ".join(g.tag_labels.get(sha, []))
        if t:
            pre.append(f"tags:{t}")

        prefix = f"[{' | '.join(pre)}] " if pre else ""
        return f"{prefix}{short}  {date_part}  {subject}".rstrip()

    def _print_tree(self, roots: List[str], children_of: Dict[str, List[str]]) -> None:
        show_files = (self.show_name_only or self.show_name_status) and (self.touched_files is not None)
        name_status = self.show_name_status

        def walk(node: str, prefix: str, is_last: bool, path: Set[str]) -> None:
            connector = "└─ " if is_last else "├─ "
            line = self._format_commit_line(node)
            if prefix:
                print(f"{prefix}{connector}{line}")
            else:
                print(line)

            if node in path:
                return
            path2 = set(path)
            path2.add(node)

            child_prefix = prefix + ("    " if is_last else "│   ")

            if show_files:
                files = self.touched_files.get(node, name_status=name_status)
                if self.max_files and len(files) > self.max_files:
                    shown = files[: self.max_files]
                    more = len(files) - self.max_files
                else:
                    shown = files
                    more = 0

                for f in shown:
                    print(f"{child_prefix}· {f}")
                if more:
                    print(f"{child_prefix}· ... ({more} more)")

            kids = children_of.get(node, [])
            for i, k in enumerate(kids):
                walk(k, child_prefix, i == len(kids) - 1, path2)

        for i, r in enumerate(roots):
            walk(r, prefix="", is_last=(i == len(roots) - 1), path=set())


# ------------------------- OO CLI -------------------------


class GitTreeCli:
    def print(
        self,
        *,
        C: str = ".",
        max_commits: int = 2000,
        selectors: RefSelectors,
        name_only: bool = False,
        name_status: bool = False,
        max_files: int = 0,
        pathspec: Optional[List[str]] = None,
    ) -> None:
        if not pathspec:
            raise RuntimeError("pathspec is required after --")
        if name_only and name_status:
            raise RuntimeError("use at most one of --name-only or --name-status")

        cwd = Path(C).expanduser().resolve()
        git = Git(cwd)
        toplevel = git.show_toplevel()

        ps = [p.replace("\\", "/") for p in pathspec]

        graph = FileHistoryGraphBuilder(git).build(
            pathspecs=ps,
            max_commits=max(1, int(max_commits)),
            selectors=selectors,
        )

        touched = CommitTouchedFiles(git, ps) if (name_only or name_status) else None

        print(f"Repo: {toplevel}")
        print(f"Pathspec: {ps}")
        print(f"HEAD branch: {graph.head_branch}")
        print(f"Touch commits: {len(graph.touch_shas)}")
        print()

        AsciiTreePrinter(
            graph,
            touched_files=touched,
            show_name_only=name_only,
            show_name_status=name_status,
            max_files=max_files,
        ).print()


# ------------------------- argparse -------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("-C", dest="C", default=".", help="run as if started in <path>")
    ap.add_argument("--max-commits", type=int, default=2000)

    # Standard git-ish ref selectors for revision sets
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--branches", action="append", nargs="?", const=_SENTINEL_ALL, default=[])
    ap.add_argument("--remotes", action="append", nargs="?", const=_SENTINEL_ALL, default=[])
    ap.add_argument("--tags", action="append", nargs="?", const=_SENTINEL_ALL, default=[])

    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--name-only", action="store_true", help="print touched paths under each commit")
    grp.add_argument("--name-status", action="store_true", help="print touched paths with status under each commit")

    ap.add_argument("--max-files", type=int, default=0, help="cap printed files per commit (0=unlimited)")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="use: -- <pathspec...>")
    ns = ap.parse_args()

    if not ns.rest or ns.rest[0] != "--":
        raise SystemExit("usage: git-tree.py [-C <path>] [opts] -- <pathspec...>")
    ns.pathspec = [p for p in ns.rest[1:] if p.strip()]
    if not ns.pathspec:
        raise SystemExit("need at least one pathspec after --")

    ns.selectors = RefSelectors(
        all=bool(ns.all),
        branches=list(ns.branches),
        remotes=list(ns.remotes),
        tags=list(ns.tags),
    )
    return ns


def main() -> None:
    ns = _parse_args()
    GitTreeCli().print(
        C=ns.C,
        max_commits=ns.max_commits,
        selectors=ns.selectors,
        name_only=ns.name_only,
        name_status=ns.name_status,
        max_files=ns.max_files,
        pathspec=ns.pathspec,
    )


if __name__ == "__main__":
    main()
