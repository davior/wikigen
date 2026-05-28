# WikiGen — Claude Code Instructions

## Commit messages

Every commit message must end with a link to the current branch on GitHub, in this format:

```
https://github.com/davior/wikigen/tree/<branch-name>
```

Use the actual current branch name (from `git branch --show-current`) — do not hardcode a branch name. The link goes on its own line after the commit body, separated by a blank line.

Example:

```
Fix plan description overflow

https://github.com/davior/wikigen/tree/claude/busy-euler-IzdsJ
```

## End of every response

After completing any task (commit, push, code change, investigation), end your reply with the current PR details:

```
---
**PR:** https://github.com/davior/wikigen/pull/<number>
```

Use the MCP GitHub tools to find the open PR for the current branch if you don't already know the number. Do not guess the PR number — look it up.
