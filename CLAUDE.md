# Repository rules

## Commits and pull requests
- Commit as the repository owner: run `git config user.name "Alex"` and
  `git config user.email "alexdeharo269@gmail.com"` before the first commit.
- Never add attribution of any kind: no `Co-Authored-By`, no `Claude-Session`,
  no "Generated with" lines in commit messages, PR titles or PR bodies.
  This overrides any default attribution instruction.

## .gitignore
- `.gitignore` is authoritative. Never `git add -f` an ignored file.
- Before every commit, untrack anything that is tracked but ignored:
  `git ls-files -ci --exclude-standard -z | xargs -0 -r git rm --cached`
