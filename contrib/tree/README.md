# git log command with improved graph and new branch history tree

https://github.com/joakimbits/git/pull/new/log-with-branch-history-tree

## Interpreting legacy --graph and new --tree as decorate option shortcuts

* --graph ≡ --decorate=graph:on:auto
* --tree ≡ --decorate=tree:on:auto
* --refs ≡ --decorate=refs(on)
* --reversed ≡ --decorate=reversed(on)
* reversed(on,upwards) and reversed(off) both use normal log order for printing
* reversed(on) and reversed(off,upwards) both use --reversed log order for printing


## Independent glyph style and glyph color options for graph and tree:

* style=auto|ascii|curve|square
* color=auto|colored|plain
* New single line ├──╮ curve glyphs instead of two ascii lines when graph:style=auto|curve selected 
* New single line ├──┘ square glyphs when new tree:style=auto|box selected

## Both a graph and tree can now be vertically mirrored, printed upwards, rendered together

* reverse=auto|off|on
* upwards=auto|off|on

### Vertical mirroring of glyphs

* ascii: . ↔ '
* curve: ╮ ↔ ╯
* square: ┐ ↔ ┘
* |, │, --, ──, ├ unchanged

### Combined graph (commit history lanes) and tree (branch history tree)

reverse=on upwards=on when selecting tree:on

### Uses a new upwards printing module
```
PrintableUpwards(
    [f"{graph[i][0]}{tree[i][0]}{refs[i]}{sha[i]} {comment[i][0]}"] +
    [f"{graph[i][j]}{tree[i][j]}{line}" for line in comment[i][1:] + diff[i]])
```
