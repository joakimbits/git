#!/usr/bin/env python3
# file: git-tree.py

"""
git-tree.py

Two renderers:

1) DAG renderer (delegates to git):
   --graph=dag

2) Tree renderer (custom):
   --graph=tree
   Builds a "branch-out tree" using a commit parent graph limited by pathspec(s)
   (or repo-wide if no pathspec is provided).

Compaction:
- --compact collapses linear non-key commits between key nodes.
- --supercompact implies --compact and ignores tag-only commits for key-node selection
  (tags still print on commits that are printed).

Touched files printing:
- --name-only / --name-status prints file changes under each printed node.
- We compute touched files by comparing the node to its *visible parent*:
    git diff [--name-only|--name-status] --no-renames <parent> <node> -- <pathspec...>
  Root nodes use:
    git diff-tree --root [--name-only|--name-status] --no-renames -r <node> -- <pathspec...>

This yields correct A/M/D summaries for compacted segments.
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
    head_tip: str
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

    def run_passthru(self, args: Sequence[str], *, timeout_s: int = 0) -> None:
        subprocess.run(
            ["git", "-C", str(self.cwd), *args],
            stdout=None,
            stderr=None,
            text=True,
            check=True,
            timeout=timeout_s if timeout_s > 0 else None,
        )

    def show_toplevel(self) -> Path:
        out = self.run(["rev-parse", "--show-toplevel"], timeout_s=30).strip()
        return Path(out)


# ------------------------- builder -------------------------


class FileHistoryGraphBuilder:
    """Builds the commit parent graph and ref labels. If pathspecs empty => repo-wide."""

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
        head_tip = self._head_tip(pathspecs)

        rows = self._log_rows(pathspecs, max_commits=max_commits, selectors=selectors)
        shas = [sha for sha, _ in rows]
        sha_set = set(shas)

        parents_map = self._ancestor_parent_graph(shas)

        nearest_cache: Dict[str, List[str]] = {}
        touch_parents: Dict[str, List[str]] = {}
        for sha, direct_parents in rows:
            parents_out: List[str] = []
            for p in direct_parents:
                parents_out.extend(
                    self._nearest_commits_in_set(
                        p,
                        sha_set=sha_set,
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

        metas = self._fetch_meta(shas)

        label_locals = True
        label_remotes = selectors.all or bool(selectors.remotes)

        branch_labels = self._branch_labels(pathspecs, include_locals=label_locals, include_remotes=label_remotes)
        tag_labels = self._tag_labels(pathspecs)

        return FileHistoryGraph(
            touch_shas=shas,
            touch_parents=touch_parents,
            metas=metas,
            branch_labels=branch_labels,
            tag_labels=tag_labels,
            head_tip=head_tip,
            head_branch=head_branch,
        )

    def _head_branch(self) -> str:
        return self.git.run(["rev-parse", "--abbrev-ref", "HEAD"], timeout_s=30).strip()

    def _head_tip(self, pathspecs: List[str]) -> str:
        if not pathspecs:
            return self.git.run(["rev-parse", "HEAD"], timeout_s=30).strip()

        out = self.git.run(
            ["log", "-n", "1", "--pretty=format:%H", "HEAD", "--", *pathspecs],
            timeout_s=120,
        ).strip()
        return out.splitlines()[0].strip() if out else ""

    def _log_rows(
        self,
        pathspecs: List[str],
        *,
        max_commits: int,
        selectors: RefSelectors,
    ) -> List[Tuple[str, List[str]]]:
        cmd = [
            "log",
            *selectors.to_git_args(),
            "--topo-order",
            f"-n{max_commits}",
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

    def _nearest_commits_in_set(
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

        out = self.git.run(
            ["log", "-n", "1", "--pretty=format:%H", ref, "--", *pathspecs],
            timeout_s=120,
        ).strip()
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


# ------------------------- touched files provider -------------------------


class CommitTouchedFiles:
    """
    Provide touched files for a node by comparing it with its *visible parent*.

    Root node:
      git diff-tree --root --no-renames -r --name-status <sha> -- <pathspec...>

    Edge parent->node:
      git diff --no-renames --name-status <parent> <sha> -- <pathspec...>
    """

    def __init__(self, git: Git, pathspecs: List[str]) -> None:
        self.git = git
        self.pathspecs = pathspecs
        self._cache: Dict[Tuple[Optional[str], str, bool], List[str]] = {}

    def get(self, sha: str, *, parent_sha: Optional[str], name_status: bool) -> List[str]:
        key = (parent_sha, sha, name_status)
        if key in self._cache:
            return self._cache[key]

        if parent_sha:
            args = [
                "diff",
                "--no-renames",
                "--name-status" if name_status else "--name-only",
                parent_sha,
                sha,
            ]
            if self.pathspecs:
                args += ["--", *self.pathspecs]
            out = self.git.run(args, timeout_s=240)
        else:
            args = [
                "diff-tree",
                "--root",
                "--no-renames",
                "-r",
                "--no-commit-id",
                "--name-status" if name_status else "--name-only",
                sha,
            ]
            if self.pathspecs:
                args += ["--", *self.pathspecs]
            out = self.git.run(args, timeout_s=240)

        lines = [ln.rstrip("\n") for ln in out.splitlines() if ln.strip()]
        self._cache[key] = lines
        return lines


# ------------------------- tree printer -------------------------


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
    """Print a rooted forest using the computed nearest-parent mapping."""

    def __init__(
        self,
        graph: FileHistoryGraph,
        *,
        style: TreeStyle,
        touched_files: Optional[CommitTouchedFiles] = None,
        show_name_only: bool = False,
        show_name_status: bool = False,
        max_files: int = 0,
        compact: bool = False,
        supercompact: bool = False,
    ) -> None:
        self.g = graph
        self.style = style
        self.touched_files = touched_files
        self.show_name_only = show_name_only
        self.show_name_status = show_name_status
        self.max_files = max(0, int(max_files))
        self.compact = bool(compact or supercompact)
        self.supercompact = bool(supercompact)

    def print(self) -> None:
        g = self.g
        if not g.touch_shas:
            print("No commits in selection.")
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

        if g.head_tip and g.head_tip in set(g.touch_shas):
            cur = g.head_tip
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
        self._print_tree(ordered_roots, children_of)

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

    def _format_commit_line(self, sha: str) -> str:
        g = self.g
        m = g.metas.get(sha)
        short = sha[:10]
        date_part = (m.date_iso[:10] if m and m.date_iso else "")
        subject = (m.subject if m else "")

        pre: List[str] = []
        if sha == g.head_tip:
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
        roots_set = set(roots)

        def print_files(prefix: str, sha: str, parent_sha: Optional[str]) -> None:
            if not show_files:
                return
            lines = self.touched_files.get(sha, parent_sha=parent_sha, name_status=name_status)

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

        def compress_chain(start: str, *, base_visible_parent: str) -> str:
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

        def walk(
            node: str,
            *,
            prefix: str,
            is_last: bool,
            path: Set[str],
            visible_parent: Optional[str],
        ) -> None:
            connector = self.style.elbow if is_last else self.style.tee
            line = self._format_commit_line(node)
            if prefix:
                print(f"{prefix}{connector}{line}")
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

            endpoints: List[str] = []
            seen_end: Set[str] = set()
            for c in kids:
                end = compress_chain(c, base_visible_parent=node)
                if end not in seen_end:
                    seen_end.add(end)
                    endpoints.append(end)

            for i, end in enumerate(endpoints):
                walk(
                    end,
                    prefix=child_prefix,
                    is_last=(i == len(endpoints) - 1),
                    path=path2,
                    visible_parent=node,  # diff against visible parent => correct segment summary
                )

        for i, r in enumerate(roots):
            walk(
                r,
                prefix="",
                is_last=(i == len(roots) - 1),
                path=set(),
                visible_parent=None,  # root: diff-tree --root
            )


# ------------------------- CLI -------------------------


class GitTreeCli:
    def run(
        self,
        *,
        C: str = ".",
        graph: str = "tree",
        graph_style: str = "unicode",
        compact: bool = False,
        supercompact: bool = False,
        reverse: bool = False,
        max_commits: int = 2000,
        selectors: RefSelectors,
        name_only: bool = False,
        name_status: bool = False,
        max_files: int = 0,
        pathspec: Optional[List[str]] = None,
    ) -> None:
        if name_only and name_status:
            raise RuntimeError("use at most one of --name-only or --name-status")

        cwd = Path(C).expanduser().resolve()
        git = Git(cwd)
        toplevel = git.show_toplevel()
        ps = [p.replace("\\", "/") for p in (pathspec or [])]

        if graph == "dag":
            if compact or supercompact:
                raise RuntimeError("--compact/--supercompact are tree-only")
            self._run_dag(
                git=git,
                toplevel=toplevel,
                selectors=selectors,
                ps=ps,
                max_commits=max_commits,
                reverse=reverse,
                name_only=name_only,
                name_status=name_status,
            )
            return

        style = TreeStyle.unicode() if graph_style == "unicode" else TreeStyle.ascii()
        graph_obj = FileHistoryGraphBuilder(git).build(
            pathspecs=ps,
            max_commits=max(1, int(max_commits)),
            selectors=selectors,
        )
        touched = CommitTouchedFiles(git, ps) if (name_only or name_status) else None

        print(f"Repo: {toplevel}")
        print(f"Pathspec: {ps if ps else '(none)'}")
        print(f"Revset: {' '.join(selectors.to_git_args()) if selectors.any() else '(default) HEAD'}")
        print(f"HEAD branch: {graph_obj.head_branch}")
        print(f"Commits: {len(graph_obj.touch_shas)}")
        if reverse:
            print("Note: --reverse is a no-op in tree mode (tree is root-first).")
        if supercompact:
            print("Mode: supercompact (tags do not affect tree shape)")
        elif compact:
            print("Mode: compact")
        print()

        TreePrinter(
            graph_obj,
            style=style,
            touched_files=touched,
            show_name_only=name_only,
            show_name_status=name_status,
            max_files=max_files,
            compact=compact,
            supercompact=supercompact,
        ).print()

    @staticmethod
    def _run_dag(
        *,
        git: Git,
        toplevel: Path,
        selectors: RefSelectors,
        ps: List[str],
        max_commits: int,
        reverse: bool,
        name_only: bool,
        name_status: bool,
    ) -> None:
        cmd = [
            "log",
            *selectors.to_git_args(),
            "--graph",
            "--oneline",
            "--decorate",
            f"-n{max(1, int(max_commits))}",
        ]
        if reverse:
            cmd.append("--reverse")
        if name_only:
            cmd.append("--name-only")
        if name_status:
            cmd.append("--name-status")
        if ps:
            cmd += ["--", *ps]

        print(f"Repo: {toplevel}")
        print(f"Revset: {' '.join(selectors.to_git_args()) if selectors.any() else '(default) HEAD'}")
        print(f"Pathspec: {ps if ps else '(none)'}")
        print()
        git.run_passthru(cmd)


# ------------------------- argparse -------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("-C", dest="C", default=".", help="run as if started in <path>")

    ap.add_argument("--graph", default="tree", choices=["dag", "tree"])
    ap.add_argument("--graph-style", default="unicode", choices=["unicode", "ascii"])

    ap.add_argument("--compact", action="store_true")
    ap.add_argument(
        "--supercompact",
        action="store_true",
        help="implies --compact; tags do not affect tree shape",
    )

    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--max-commits", type=int, default=2000)

    ap.add_argument("--all", action="store_true")
    ap.add_argument("--branches", action="append", nargs="?", const=_SENTINEL_ALL, default=[])
    ap.add_argument("--remotes", action="append", nargs="?", const=_SENTINEL_ALL, default=[])
    ap.add_argument("--tags", action="append", nargs="?", const=_SENTINEL_ALL, default=[])

    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--name-only", action="store_true")
    grp.add_argument("--name-status", action="store_true")
    ap.add_argument("--max-files", type=int, default=0, help="0=unlimited")

    ap.add_argument("rest", nargs=argparse.REMAINDER, help="optional: -- <pathspec...>")
    ns = ap.parse_args()

    ns.pathspec = []
    if ns.rest:
        if ns.rest[0] == "--":
            ns.pathspec = [p for p in ns.rest[1:] if p.strip()]
        else:
            raise SystemExit("pathspecs must follow `--` (or omit `--` entirely for none)")

    ns.selectors = RefSelectors(
        all=bool(ns.all),
        branches=list(ns.branches),
        remotes=list(ns.remotes),
        tags=list(ns.tags),
    )
    return ns


def main() -> None:
    ns = _parse_args()
    GitTreeCli().run(
        C=ns.C,
        graph=ns.graph,
        graph_style=ns.graph_style,
        compact=ns.compact,
        supercompact=ns.supercompact,
        reverse=ns.reverse,
        max_commits=ns.max_commits,
        selectors=ns.selectors,
        name_only=ns.name_only,
        name_status=ns.name_status,
        max_files=ns.max_files,
        pathspec=ns.pathspec,
    )


if __name__ == "__main__":
    main()
