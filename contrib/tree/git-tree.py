#!/usr/bin/env python3
# file: git-tree.py
"""
Contrib prototype: render a file-history "tree" view plus a "dag" passthrough view.

Key behaviors:
- Tree view reuses git's own commit-line formatting (oneline/pretty/format/decorate) via `git show -s`.
- Captured git output forces color like `git log --color=auto` would on a TTY.
- Synthetic touch-tip decorations are only injected when git's decoration (%d) is empty.
- supercompact: prefer local branch over remote for synthetic touch-tip refs.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ---------------------------- git runner (crash on failure) ----------------------------


class Git:
    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def run(self, args: Sequence[str], *, timeout_s: int = 120, stdin_text: Optional[str] = None) -> str:
        cp = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            input=stdin_text,
            stdout=subprocess.PIPE,
            stderr=None,  # let git print to stderr on crash
            text=True,
            check=True,
            timeout=timeout_s,
        )
        return cp.stdout

    def run_bytes(self, args: Sequence[str], *, timeout_s: int = 120, stdin_bytes: Optional[bytes] = None) -> bytes:
        cp = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            input=stdin_bytes,
            stdout=subprocess.PIPE,
            stderr=None,
            check=True,
            timeout=timeout_s,
        )
        return cp.stdout

    def run_passthru(self, args: Sequence[str]) -> None:
        subprocess.run(
            ["git", "-C", str(self.repo), *args],
            stdout=None,
            stderr=None,
            check=True,
        )


def _git_env_no_pager() -> Dict[str, str]:
    env = os.environ.copy()
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    return env


# ---------------------------- argv split / passthrough ----------------------------


def _split_git_and_pathspec(rest: List[str]) -> Tuple[List[str], List[str]]:
    if "--" in rest:
        i = rest.index("--")
        return rest[:i], [p for p in rest[i + 1 :] if p.strip()]
    return rest, []


def _contains_git_graph_arg(args: List[str]) -> bool:
    return any(a == "--graph" or a.startswith("--graph=") for a in args)


def _user_color_choice(git_args: List[str]) -> Optional[str]:
    # explicit overrides only
    for a in git_args:
        if a in ("--color=always", "--color=never", "--color=auto"):
            return a.split("=", 1)[1]
        if a == "--color":  # rarely used like "--color always"
            # argparse-like form; we won't parse the next token here
            return None
    return None


def _color_arg_for_captured(git_args: List[str]) -> str:
    """
    Mimic git's --color=auto semantics for *captured* stdout (PIPE):
      - if user says --color=never -> never
      - else if stdout is a tty -> always
      - else -> never
    """
    choice = _user_color_choice(git_args)
    if choice == "never":
        return "--color=never"
    if choice == "always":
        return "--color=always"
    # auto or unspecified
    return "--color=always" if sys.stdout.isatty() else "--color=never"


def _extract_decorate_arg(git_args: List[str]) -> Optional[str]:
    decorate = None
    for a in git_args:
        if a == "--decorate" or a.startswith("--decorate="):
            decorate = a
        if a == "--no-decorate":
            decorate = "--decorate=no"
    return decorate


def _drop_color_args(args: List[str]) -> List[str]:
    out: List[str] = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a == "--color":
            skip_next = True
            continue
        if a.startswith("--color="):
            continue
        out.append(a)
    return out


def _drop_graph_args(args: List[str]) -> List[str]:
    return [a for a in args if a != "--graph" and not a.startswith("--graph=")]


def _extract_abbrev_len(git_args: List[str]) -> int:
    if "--no-abbrev-commit" in git_args:
        return 40
    for a in git_args:
        if a.startswith("--abbrev="):
            try:
                n = int(a.split("=", 1)[1])
                return max(1, min(40, n))
            except ValueError:
                pass
    return 7 if "--oneline" in git_args else 10


def _has_any_pretty(git_args: List[str]) -> bool:
    return any(
        a == "--oneline"
        or a.startswith("--pretty")
        or a.startswith("--format")
        or a in ("--abbrev-commit", "--no-abbrev-commit")
        for a in git_args
    )


def _pretty_passthrough_args(git_args: List[str]) -> List[str]:
    """
    Keep only commit-line formatting args that `git show -s` understands and that
    affect what user expects to see in `git log`.
    """
    keep_prefixes = (
        "--pretty",
        "--format",
        "--date=",
        "--abbrev=",
        "--decorate",
        "--no-decorate",
    )
    keep_exact = {
        "--oneline",
        "--abbrev-commit",
        "--no-abbrev-commit",
    }
    out: List[str] = []
    for a in git_args:
        if a in keep_exact:
            out.append(a)
            continue
        if any(a.startswith(p) for p in keep_prefixes):
            out.append(a)
            continue
    return out


# ---------------------------- ANSI helpers ----------------------------

_ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _strip_ansi(s: str) -> str:
    return _ANSI_CSI_RE.sub("", s)


# ---------------------------- touch-tip synthetic decoration ----------------------------


def _prefer_local_over_remote(refs: List[str], local_branches: Set[str]) -> List[str]:
    out: List[str] = []
    for r in refs:
        if "/" in r:
            _, tail = r.split("/", 1)
            if tail in local_branches:
                continue
        out.append(r)
    return out


def _format_synthetic_decorations(
    *,
    sha: str,
    head_tip: str,
    head_branch: str,
    touch_refs: List[str],
    touch_tags: List[str],
    supercompact: bool,
    local_branches: Set[str],
) -> str:
    parts: List[str] = []
    if sha == head_tip and head_branch and head_branch != "HEAD":
        parts.append(f"HEAD -> {head_branch}")

    refs = touch_refs
    if supercompact:
        refs = _prefer_local_over_remote(refs, local_branches)
    parts.extend(refs)

    # supercompact: tags don't affect tree shape and are typically too noisy
    if not supercompact:
        parts.extend([f"tag: {t}" for t in touch_tags])

    return "(" + ", ".join(parts) + ")" if parts else ""


def _git_line_has_decoration(line: str) -> bool:
    """
    Heuristic: git --decorate puts "(...)" immediately after the hash (or after hash+space).
    We detect: "<hash> (" in the stripped line.
    """
    s = _strip_ansi(line).lstrip()
    if not s:
        return False
    # first token is hash (abbrev or full)
    parts = s.split(" ", 1)
    if len(parts) != 2:
        return False
    rest = parts[1].lstrip()
    return rest.startswith("(")


def _inject_decoration_into_git_line(line: str, decoration: str) -> str:
    """
    Insert "(...)" after the first token (commit hash) if there's no existing decoration.
    """
    if not decoration:
        return line
    if _git_line_has_decoration(line):
        return line
    s = line.lstrip()
    leading = line[: len(line) - len(s)]
    parts = s.split(" ", 1)
    if len(parts) == 1:
        return f"{leading}{parts[0]} {decoration}"
    h, rest = parts
    rest = rest.lstrip()
    return f"{leading}{h} {decoration} {rest}".rstrip()


# ---------------------------- core graph model ----------------------------


@dataclass(frozen=True)
class HistoryGraph:
    shas: List[str]                      # commits in selection
    parents: Dict[str, List[str]]        # compressed to nearest-in-set parents
    children: Dict[str, List[str]]       # derived

    # user-visible commit lines rendered by git itself (may be multi-line)
    rendered_lines: Dict[str, List[str]]  # sha -> [line0, line1, ...]

    # touch-tip labels (your file-history concept)
    touch_branch_labels: Dict[str, List[str]]  # sha -> [refname:short]
    touch_tag_labels: Dict[str, List[str]]     # sha -> [tagname]
    local_branches: Set[str]

    # HEAD info
    head_tip: str
    head_branch: str

    # config-ish
    abbrev_len: int
    decorate_requested: bool


def _chunked(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


class GraphBuilder:
    def __init__(self, git: Git) -> None:
        self.git = git

    def build(
        self,
        *,
        git_log_args: List[str],
        pretty_args: List[str],
        pathspecs: List[str],
        max_commits: int,
        decorate_requested: bool,
        abbrev_len: int,
        captured_color_arg: str,
    ) -> HistoryGraph:
        head_branch = self.git.run(["rev-parse", "--abbrev-ref", "HEAD"], timeout_s=30).strip()
        head_tip = self._head_tip(pathspecs)

        rows = self._log_rows(git_log_args=git_log_args, pathspecs=pathspecs, max_commits=max_commits)
        shas = [sha for sha, _ in rows]
        sha_set = set(shas)

        parents_map = self._ancestor_parent_graph(shas)

        nearest_cache: Dict[str, List[str]] = {}
        parents: Dict[str, List[str]] = {}
        for sha, direct_parents in rows:
            out: List[str] = []
            for p in direct_parents:
                out.extend(self._nearest_in_set(p, sha_set=sha_set, parents_map=parents_map, cache=nearest_cache))
            uniq: List[str] = []
            seen: Set[str] = set()
            for x in out:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            parents[sha] = uniq

        children: Dict[str, List[str]] = {sha: [] for sha in shas}
        for child, ps in parents.items():
            for p in ps:
                if p in children:
                    children[p].append(child)

        # stable-ish ordering by commit time (via git show)
        epoch = self._epochs(shas)
        for p, kids in children.items():
            kids.sort(key=lambda s: epoch.get(s, 0), reverse=True)

        local_branches = set(self._refs_list(include_locals=True, include_remotes=False))
        touch_br = self._touch_branch_labels(pathspecs, include_locals=True, include_remotes=True)
        touch_tags = self._touch_tag_labels(pathspecs)

        rendered_lines = self._render_commit_lines(
            shas,
            pretty_args=pretty_args,
            pathspecs=pathspecs,
            captured_color_arg=captured_color_arg,
        )

        return HistoryGraph(
            shas=shas,
            parents=parents,
            children=children,
            rendered_lines=rendered_lines,
            touch_branch_labels=touch_br,
            touch_tag_labels=touch_tags,
            local_branches=local_branches,
            head_tip=head_tip,
            head_branch=head_branch,
            abbrev_len=abbrev_len,
            decorate_requested=decorate_requested,
        )

    def _epochs(self, shas: List[str]) -> Dict[str, int]:
        if not shas:
            return {}
        fmt = "%H%x1f%ct%x1e"
        out: Dict[str, int] = {}
        for chunk in _chunked(list(dict.fromkeys(shas)), 200):
            txt = self.git.run(["show", "-s", f"--pretty=format:{fmt}", *chunk], timeout_s=240)
            for rec in txt.split("\x1e"):
                rec = rec.strip()
                if not rec:
                    continue
                parts = rec.split("\x1f")
                if len(parts) != 2:
                    continue
                sha, ct = parts[0].strip(), parts[1].strip()
                try:
                    out[sha] = int(ct)
                except ValueError:
                    out[sha] = 0
        return out

    def _render_commit_lines(
        self,
        shas: List[str],
        *,
        pretty_args: List[str],
        pathspecs: List[str],
        captured_color_arg: str,
    ) -> Dict[str, List[str]]:
        """
        Render each commit using git itself, preserving user's pretty/oneline/decorate output.
        We record the full output per sha (can be multi-line).
        """
        if not shas:
            return {}

        # Ensure single-line default (match git log --oneline-ish) if user provided no pretty options.
        pretty = list(pretty_args)
        if not any(a == "--oneline" or a.startswith("--pretty") or a.startswith("--format") for a in pretty):
            pretty = ["--oneline", *pretty]

        # Ensure we only get that commit's formatted output (no patch)
        # Note: `git show -s` prints a trailing newline; we splitlines().
        out: Dict[str, List[str]] = {}
        for chunk in _chunked(list(dict.fromkeys(shas)), 100):
            # Use a record separator between commits so we can map back.
            # We'll ask git to prepend the full SHA in a hidden field using pretty format,
            # but only when user did NOT specify their own --pretty/--format.
            user_pretty = any(a.startswith("--pretty") or a.startswith("--format") for a in pretty_args)
            if user_pretty:
                # Can't safely wrap arbitrary user format; do per-commit calls.
                for sha in chunk:
                    txt = self.git.run(
                        ["show", "-s", "--no-patch", captured_color_arg, *pretty, sha],
                        timeout_s=240,
                    )
                    lines = [ln.rstrip("\n") for ln in txt.splitlines() if ln.strip() != ""]
                    out[sha] = lines if lines else [sha[:10]]
                continue

            # Safe path: we control the format: emit "<SHA><US><USER_OUTPUT><RS>"
            RS = "\x1e"
            US = "\x1f"
            # user output: oneline-ish by default; allow decorate etc via pretty args
            # We'll approximate oneline with: "%h %d %s" if user passed --oneline (git will do it)
            # so we simply use the pretty args and let git format.
            fmt = f"%H{US}%h %d %s{RS}"
            txt = self.git.run(
                ["show", "-s", "--no-patch", captured_color_arg, f"--pretty=format:{fmt}", *[a for a in pretty if a != "--oneline"], *chunk],
                timeout_s=240,
            )
            for rec in txt.split(RS):
                rec = rec.strip()
                if not rec:
                    continue
                if US not in rec:
                    continue
                sha, line = rec.split(US, 1)
                sha = sha.strip()
                line = line.rstrip("\n")
                out[sha] = [line]

        return out

    def _head_tip(self, pathspecs: List[str]) -> str:
        if not pathspecs:
            return self.git.run(["rev-parse", "HEAD"], timeout_s=30).strip()
        out = self.git.run(["log", "-n", "1", "--pretty=format:%H", "HEAD", "--", *pathspecs], timeout_s=120).strip()
        return out.splitlines()[0].strip() if out else ""

    def _log_rows(self, *, git_log_args: List[str], pathspecs: List[str], max_commits: int) -> List[Tuple[str, List[str]]]:
        cmd = ["log", *git_log_args, "--topo-order", f"-n{max_commits}", "--no-patch", "--pretty=format:%H %P"]
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

    def _touch_branch_labels(
        self,
        pathspecs: List[str],
        *,
        include_locals: bool,
        include_remotes: bool,
    ) -> Dict[str, List[str]]:
        commit_to_refs: Dict[str, List[str]] = {}
        for ref in self._refs_list(include_locals=include_locals, include_remotes=include_remotes):
            tip = self._tip_sha(ref, pathspecs)
            if tip:
                commit_to_refs.setdefault(tip, []).append(ref)
        for sha in commit_to_refs:
            commit_to_refs[sha].sort(key=str.casefold)
        return commit_to_refs

    def _touch_tag_labels(self, pathspecs: List[str]) -> Dict[str, List[str]]:
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


# ---------------------------- touched files (optional) ----------------------------


class EdgeTouchedFiles:
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
            cmd = ["diff", *self.diff_args, "--name-status" if name_status else "--name-only", parent_sha, sha]
        else:
            cmd = ["diff-tree", "--root", "-r", "--no-commit-id", *self.diff_args, "--name-status" if name_status else "--name-only", sha]

        if self.pathspecs:
            cmd += ["--", *self.pathspecs]

        out = self.git.run(cmd, timeout_s=240)
        lines = [ln.rstrip("\n") for ln in out.splitlines() if ln.strip()]
        self._cache[key] = lines
        return lines


def _partition_args_for_tree_mode(git_args: List[str]) -> Tuple[List[str], List[str], bool]:
    """
    Return (log_args, diff_args, show_files)
    Only support --name-only/--name-status; other args pass to log (except formatting args handled separately).
    """
    name_only = "--name-only" in git_args
    name_status = "--name-status" in git_args
    diff_args: List[str] = []
    log_args: List[str] = []
    for a in git_args:
        if a in ("--name-only", "--name-status"):
            continue
        if a.startswith("--diff-filter=") or a.startswith("--find-renames") or a.startswith("--find-copies") or a.startswith("--unified=") or a.startswith("-U") or a in ("-M", "-C", "--no-renames"):
            diff_args.append(a)
        else:
            log_args.append(a)
    return log_args, diff_args, bool(name_status or name_only)


# ---------------------------- tree rendering ----------------------------


@dataclass(frozen=True)
class TreeStyle:
    tee: str
    elbow: str
    vert: str
    space: str

    @staticmethod
    def box() -> "TreeStyle":
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
        show_files: bool,
        name_status: bool,
        max_files: int,
    ) -> None:
        self.g = g
        self.style = style
        self.compact = bool(compact or supercompact)
        self.supercompact = bool(supercompact)
        self.touched = touched
        self.show_files = bool(show_files)
        self.name_status = bool(name_status)
        self.max_files = max(0, int(max_files))

    def _is_leaf(self, sha: str) -> bool:
        return len(self.g.children.get(sha, [])) == 0

    def _is_branch_point(self, sha: str) -> bool:
        return len(self.g.children.get(sha, [])) >= 2

    def _is_key_node(self, sha: str, roots: Set[str]) -> bool:
        if sha in roots:
            return True
        if self._is_leaf(sha):
            return True
        if self._is_branch_point(sha):
            return True
        if sha == self.g.head_tip:
            return True
        if sha in self.g.touch_branch_labels:
            return True
        if (not self.supercompact) and sha in self.g.touch_tag_labels:
            return True
        return False

    def _compress_chain(self, start: str, roots: Set[str]) -> str:
        if not self.compact:
            return start
        cur = start
        while True:
            if self._is_key_node(cur, roots):
                return cur
            kids = self.g.children.get(cur, [])
            if len(kids) != 1:
                return cur
            cur = kids[0]

    def _fmt_commit_lines(self, sha: str) -> List[str]:
        g = self.g
        base = g.rendered_lines.get(sha) or [sha[: g.abbrev_len]]

        # Inject synthetic touch-tip decoration ONLY if git produced no "(...)" on line0.
        touch_refs = g.touch_branch_labels.get(sha, [])
        touch_tags = g.touch_tag_labels.get(sha, [])
        syn = _format_synthetic_decorations(
            sha=sha,
            head_tip=g.head_tip,
            head_branch=g.head_branch,
            touch_refs=touch_refs,
            touch_tags=touch_tags,
            supercompact=self.supercompact,
            local_branches=g.local_branches,
        )
        if syn and not _git_line_has_decoration(base[0]):
            base = [_inject_decoration_into_git_line(base[0], syn), *base[1:]]
        return base

    def render(self) -> str:
        g = self.g
        if not g.shas:
            return "No commits in selection.\n"

        roots: List[str] = [sha for sha in g.shas if not g.parents.get(sha)]
        # show older roots first
        roots.reverse()
        roots_set = set(roots)

        out: List[str] = []

        def print_files(prefix: str, sha: str, parent_sha: Optional[str]) -> None:
            if not (self.show_files and self.touched):
                return
            lines = self.touched.get(sha, parent_sha=parent_sha, name_status=self.name_status)
            if self.max_files and len(lines) > self.max_files:
                shown = lines[: self.max_files]
                more = len(lines) - self.max_files
            else:
                shown = lines
                more = 0
            for ln in shown:
                out.append(f"{prefix}· {ln}\n")
            if more:
                out.append(f"{prefix}· ... ({more} more)\n")

        def walk(node: str, *, prefix: str, is_last: bool, path: Set[str], visible_parent: Optional[str]) -> None:
            conn = self.style.elbow if is_last else self.style.tee
            lines = self._fmt_commit_lines(node)

            if prefix:
                out.append(f"{prefix}{conn}{lines[0]}\n")
            else:
                out.append(f"{lines[0]}\n")

            if len(lines) > 1:
                cont_prefix = prefix + (self.style.space if is_last else self.style.vert)
                for extra in lines[1:]:
                    out.append(f"{cont_prefix}{extra}\n")

            if node in path:
                return
            path2 = set(path)
            path2.add(node)

            child_prefix = prefix + (self.style.space if is_last else self.style.vert)
            print_files(child_prefix, node, visible_parent)

            kids = g.children.get(node, [])
            if not kids:
                return

            nexts: List[str] = []
            seen: Set[str] = set()
            for k in kids:
                end = self._compress_chain(k, roots_set)
                if end not in seen:
                    seen.add(end)
                    nexts.append(end)

            for i, ch in enumerate(nexts):
                walk(ch, prefix=child_prefix, is_last=(i == len(nexts) - 1), path=path2, visible_parent=node)

        for i, r in enumerate(roots):
            walk(r, prefix="", is_last=(i == len(roots) - 1), path=set(), visible_parent=None)

        return "".join(out)


# ---------------------------- dag glyph conversion (keep rest colors) ----------------------------

_GRAPH_CHARS = set(r"|/\*_. -<>o=^+-")
_WIRE_MAP_DAG = str.maketrans({"|": "│", "/": "╱", "\\": "╲", "_": "─", "-": "─", ".": "·"})
_MERGE_ASCII = "|\\ "
_FORK_ASCII = "|/ "
_MERGE_BOX = "├─┐"
_FORK_BOX = "├─┘"


def _split_prefix_ansi_aware(line: str) -> Tuple[str, str]:
    i = 0
    n = len(line)
    while i < n:
        m = _ANSI_CSI_RE.match(line, i)
        if m:
            i = m.end()
            continue
        if line[i] in _GRAPH_CHARS:
            i += 1
            continue
        break
    return line[:i], line[i:]


def _is_graph_only(prefix_plain: str, rest: str) -> bool:
    return rest.strip() == "" and prefix_plain.strip() != "" and "*" not in prefix_plain


def _find_overlays(prefix_plain: str) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    prev: List[Tuple[int, str]] = []
    nxt: List[Tuple[int, str]] = []

    start = 0
    while True:
        p = prefix_plain.find(_FORK_ASCII, start)
        if p < 0:
            break
        prev.append((p, _FORK_BOX))
        start = p + 1

    start = 0
    while True:
        p = prefix_plain.find(_MERGE_ASCII, start)
        if p < 0:
            break
        nxt.append((p, _MERGE_BOX))
        start = p + 1

    return prev, nxt


def _rewrite_commit_prefix(prefix_plain: str, overlays: List[Tuple[int, str]]) -> str:
    vis = prefix_plain.translate(_WIRE_MAP_DAG)
    i = vis.find("*")
    if i >= 0:
        vis = vis[:i] + "∙" + vis[i + 1 :]
    chars = list(vis)
    for pos, repl in overlays:
        if 0 <= pos and pos + 3 <= len(chars):
            chars[pos : pos + 3] = list(repl)
    return "".join(chars)


@dataclass
class _DagBuf:
    prefix_plain: str
    rest: str
    overlays: List[Tuple[int, str]]


def graph_box_glyphs_keep_rest_colors(text: str) -> str:
    out: List[str] = []
    buf: Optional[_DagBuf] = None
    pending_for_next: List[Tuple[int, str]] = []
    pending_for_prev: List[Tuple[int, str]] = []

    for line in text.splitlines(True):
        prefix, rest = _split_prefix_ansi_aware(line)
        prefix_plain = _strip_ansi(prefix)

        if _is_graph_only(prefix_plain, rest.strip("\n")):
            prev, nxt = _find_overlays(prefix_plain)
            pending_for_prev.extend(prev)
            pending_for_next.extend(nxt)
            continue

        if buf is not None:
            buf.overlays.extend(pending_for_prev)
            pending_for_prev = []
            out.append(_rewrite_commit_prefix(buf.prefix_plain, buf.overlays) + buf.rest)
            buf = None

        buf = _DagBuf(prefix_plain=prefix_plain, rest=rest, overlays=list(pending_for_next))
        pending_for_next = []

    if buf is not None:
        buf.overlays.extend(pending_for_prev)
        out.append(_rewrite_commit_prefix(buf.prefix_plain, buf.overlays) + buf.rest)

    return "".join(out)


# ---------------------------- CLI ----------------------------

def _run(
    *,
    repo: Path,
    view: str,
    graph_glyphs: str,
    tree_style: str,
    compact: bool,
    supercompact: bool,
    max_files: int,
    git_args: List[str],
    pathspecs: List[str],
) -> None:
    repo = repo.expanduser().resolve()
    git = Git(repo)

    pathspecs = [p.replace("\\", "/") for p in pathspecs]

    captured_color_arg = _color_arg_for_captured(git_args)
    decorate_arg = _extract_decorate_arg(git_args)
    decorate_requested = bool(decorate_arg and decorate_arg != "--decorate=no")
    abbrev_len = _extract_abbrev_len(git_args)

    if view == "dag":
        # passthrough unless we rewrite glyphs
        if graph_glyphs == "box" and _contains_git_graph_arg(git_args):
            git_args2 = _drop_color_args(git_args)
            cmd = ["--no-pager", "log", captured_color_arg, *git_args2]
            if pathspecs:
                cmd += ["--", *pathspecs]
            out = git.run(cmd, timeout_s=600)
            out = graph_box_glyphs_keep_rest_colors(out)
            sys.stdout.write(out)
            return

        cmd = ["log", *git_args]
        if pathspecs:
            cmd += ["--", *pathspecs]
        git.run_passthru(cmd)
        return

    # tree view
    if _contains_git_graph_arg(git_args):
        raise RuntimeError("git-tree: use --view=dag for git's --graph")

    # allow --name-only / --name-status to work (optional)
    log_args, diff_args, show_files = _partition_args_for_tree_mode(git_args)
    name_status = "--name-status" in git_args
    touched = EdgeTouchedFiles(git, pathspecs, diff_args) if show_files else None

    # tree building should not inherit pretty/format flags (we render separately)
    log_args = [a for a in _drop_color_args(_drop_graph_args(log_args)) if not (a == "--oneline" or a.startswith("--pretty") or a.startswith("--format") or a.startswith("--date=") or a.startswith("--abbrev=") or a in ("--abbrev-commit", "--no-abbrev-commit") or a.startswith("--decorate") or a == "--no-decorate")]

    pretty_args = _pretty_passthrough_args(git_args)

    max_commits = 2000
    for i, a in enumerate(git_args):
        if a == "-n" and i + 1 < len(git_args):
            try:
                max_commits = int(git_args[i + 1])
            except ValueError:
                pass
        if a.startswith("--max-count="):
            try:
                max_commits = int(a.split("=", 1)[1])
            except ValueError:
                pass

    g = GraphBuilder(git).build(
        git_log_args=log_args,
        pretty_args=pretty_args,
        pathspecs=pathspecs,
        max_commits=max(1, int(max_commits)),
        decorate_requested=decorate_requested,
        abbrev_len=abbrev_len,
        captured_color_arg=captured_color_arg,
    )

    style = TreeStyle.ascii() if tree_style == "ascii" else TreeStyle.box()
    text = TreePrinter(
        g,
        style=style,
        compact=compact,
        supercompact=supercompact,
        touched=touched,
        show_files=show_files,
        name_status=name_status,
        max_files=max_files,
    ).render()
    sys.stdout.write(text)

# -------------------------
# Typer CLI (prototype)
# -------------------------

from dataclasses import field
import typer

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)

CLASSES = {"graph", "tree", "refs", "reversed"}
MODES = {"auto", "on", "off"}


@dataclass
class DecoratorSpec:
    kind: str
    mode: str = "auto"
    style: Optional[str] = None
    color: str = "auto"
    extras: List[str] = field(default_factory=list)


def parse_spec_list(value: str) -> List[DecoratorSpec]:
    """
    Parse:  spec-list := spec (',' spec)*
            spec      := CLASS (':' field)*
            field     := VALUE | KEY '=' VALUE
    """
    specs: List[DecoratorSpec] = []
    for raw_spec in [s.strip() for s in value.split(",") if s.strip()]:
        parts = [p.strip() for p in raw_spec.split(":") if p.strip()]
        if not parts:
            continue

        kind = parts[0]
        if kind not in CLASSES:
            raise typer.BadParameter(
                f"Unknown decorator class '{kind}'. Expected one of {sorted(CLASSES)}"
            )

        spec = DecoratorSpec(kind=kind)

        positional: List[str] = []
        for field_ in parts[1:]:
            if "=" in field_:
                k, v = field_.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k == "mode":
                    spec.mode = v
                elif k == "style":
                    spec.style = v
                elif k == "color":
                    spec.color = v
                else:
                    spec.extras.append(f"{k}={v}")
            else:
                positional.append(field_)

        if positional:
            if spec.mode == "auto" and positional and positional[0] in MODES:
                spec.mode = positional.pop(0)

            if positional:
                spec.style = positional.pop(0)

            if positional and positional[0] in MODES:
                spec.color = positional.pop(0)

            spec.extras.extend(positional)

        if spec.mode not in MODES:
            raise typer.BadParameter(f"{kind}: invalid mode '{spec.mode}' (expected auto|on|off)")
        if spec.color not in MODES:
            raise typer.BadParameter(f"{kind}: invalid color '{spec.color}' (expected auto|on|off)")

        specs.append(spec)

    return specs


def _merge_last_wins(specs: List[DecoratorSpec]) -> Dict[str, DecoratorSpec]:
    final: Dict[str, DecoratorSpec] = {}
    for s in specs:
        final[s.kind] = s
    return final


def _ensure_flag(args: List[str], flag: str) -> None:
    if flag not in args:
        args.append(flag)


def _drop_flags_prefix(args: List[str], *, prefixes: Tuple[str, ...], exact: Tuple[str, ...]) -> List[str]:
    out: List[str] = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a in exact:
            if a in ("--decorate", "--color"):
                skip_next = True
            continue
        if any(a.startswith(p) for p in prefixes):
            continue
        out.append(a)
    return out


def _apply_decorators(
    final: Dict[str, DecoratorSpec],
    *,
    git_args: List[str],
    default_view: str,
) -> Tuple[str, str, str, bool, bool, int]:
    """
    Returns: (view, graph_glyphs, tree_style, compact, supercompact, max_files)

    Mutates git_args to inject git flags (e.g. --graph, --decorate, --reverse, --color=...).
    """
    view = default_view
    graph_glyphs = "ascii"
    tree_style = "box"
    compact = False
    supercompact = False
    max_files = 0

    # tree decorator
    t = final.get("tree")
    if t and t.mode == "off":
        view = "dag"
    elif t and t.mode in ("on", "auto"):
        view = "tree"
        if t.style in ("ascii", "box"):
            tree_style = t.style
        for x in t.extras:
            if x == "compact":
                compact = True
            elif x == "supercompact":
                supercompact = True
            elif x.startswith("max_files="):
                try:
                    max_files = int(x.split("=", 1)[1])
                except ValueError:
                    pass

    # graph decorator
    g = final.get("graph")
    if g and g.mode == "on":
        view = "dag"
        if g.style in ("ascii", "box"):
            graph_glyphs = g.style
        _ensure_flag(git_args, "--graph")

    # refs decorator -> git --decorate/--no-decorate
    r = final.get("refs")
    if r:
        git_args[:] = _drop_flags_prefix(
            git_args,
            prefixes=("--decorate=",),
            exact=("--decorate", "--no-decorate"),
        )
        if r.mode == "off":
            git_args.append("--no-decorate")
        elif r.mode in ("on", "auto"):
            if r.style:
                git_args.append(f"--decorate={r.style}")
            else:
                git_args.append("--decorate")

    # reversed decorator -> git --reverse
    rv = final.get("reversed")
    if rv:
        git_args[:] = [a for a in git_args if a != "--reverse"]
        if rv.mode == "on":
            git_args.append("--reverse")

    # color hint (only if user didn't already pass --color)
    forced_color: Optional[str] = None
    for s in final.values():
        if s.color == "on":
            forced_color = "always"
        elif s.color == "off":
            forced_color = "never"
    if forced_color and _user_color_choice(git_args) is None:
        git_args.append(f"--color={forced_color}")

    return view, graph_glyphs, tree_style, compact, supercompact, max_files


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "allow_interspersed_args": False,
    }
)
def main(
    ctx: typer.Context,
    repo: Path = typer.Option(
        Path("."),
        "-C",
        "--repo",
        help="Run as if git was started in <path> (like git -C).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    decorate: List[str] = typer.Option(
        None,
        "--decorate",
        help=(
            "Repeatable. Comma-separated specs like graph:on:ascii,refs:on:short "
            "or graph:mode=on:style=box."
        ),
        callback=lambda v: v,
    ),
    graph: bool = typer.Option(False, "--graph", help="Alias for --decorate graph:on"),
    tree: bool = typer.Option(False, "--tree", help="Alias for --decorate tree:on"),
    refs: bool = typer.Option(False, "--refs", help="Alias for --decorate refs:on"),
    reversed_: bool = typer.Option(False, "--reversed", help="Alias for --decorate reversed:on"),
) -> None:
    """
    Prototype wrapper around `git log` that can render a tree view.

    - Wrapper options must appear before the first positional arg (e.g. before `log`).
    - Use `--` to separate git-log args from pathspecs (paths). Only args after `--` are treated as pathspecs.
    """
    raw: List[str] = decorate or []
    if graph:
        raw.append("graph:on")
    if tree:
        raw.append("tree:on")
    if refs:
        raw.append("refs:on")
    if reversed_:
        raw.append("reversed:on")

    specs: List[DecoratorSpec] = []
    for s in raw:
        specs.extend(parse_spec_list(s))
    final = _merge_last_wins(specs)

    # Everything not consumed by Typer is forwarded to git log.
    git_args = list(ctx.args)

    # Prototype CLI: `git-tree ... log [git-log-args...] [-- pathspecs...]`
    default_view = "tree"
    if git_args and git_args[0] == "log":
        default_view = "dag"
        git_args = git_args[1:]

        # Prototype log options (must appear after `log`):
        # - --tree: switch to tree renderer (do NOT forward to git log)
        if "--tree" in git_args:
            git_args = [a for a in git_args if a != "--tree"]
            default_view = "tree"
            final["tree"] = DecoratorSpec(kind="tree", mode="on")

        # Defensive: Click may leave literal '--' in ctx.args in some edge cases.
        git_args = [a for a in git_args if a != "--"]

    # Strict pathspecs: only after `--` (Click strips it, so inspect sys.argv)
    argv = sys.argv[1:]
    pathspecs: List[str] = []
    if "--" in argv:
        sep = argv.index("--")
        pathspecs = argv[sep + 1 :]
        if pathspecs and len(git_args) >= len(pathspecs) and git_args[-len(pathspecs) :] == pathspecs:
            git_args = git_args[: -len(pathspecs)]
    # Defensive: ensure we don't forward a stray separator to git (prevents `-- -- pathspec`).
    git_args = [a for a in git_args if a != "--"]

    view, graph_glyphs, tree_style, compact, supercompact, max_files = _apply_decorators(
        final,
        git_args=git_args,
        default_view=default_view,
    )

    _run(
        repo=repo,
        view=view,
        graph_glyphs=graph_glyphs,
        tree_style=tree_style,
        compact=compact,
        supercompact=supercompact,
        max_files=max_files,
        git_args=git_args,
        pathspecs=pathspecs,
    )


if __name__ == "__main__":
    # Bubble exceptions for normal Python tracebacks (debug-friendly)
    app(standalone_mode=False)
