# Issue Tracking

This project now uses [Beads (bd)](https://github.com/steveyegge/beads) for issue tracking.

## Quick Commands

```bash
bd ready                              # Show issues ready to work
bd list --status=open                 # List all open issues
bd update <id> --status=in_progress  # Claim work
bd close <id>                         # Mark complete
```

For complete workflow documentation, see `.cursor/rules/beads.mdc`

---

## Recently Completed ✓

- Add delete/rename functionality for runs, queries, and prompts in UI
- Split evals directory into evals-engineer and evals-consultant
- Make evals-dir configurable via --evals-dir argument
- Convert config.py to config.json
- Improve error propagation from server to UI
- Incorporate graph.html as iframe in index.html
