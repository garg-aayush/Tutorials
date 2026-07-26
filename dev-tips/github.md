# GitHub

## Check your permissions on a repo

```bash
gh repo view <org>/<repo> --json viewerPermission
```

Returns one of: `READ`, `TRIAGE`, `WRITE`, `MAINTAIN`, `ADMIN`.

## Pull request vs draft pull request

Mechanically the same GitHub object. A draft just carries a "not ready to merge" flag that changes a few behaviours.

| | **Draft PR** | **Regular PR** |
|---|---|---|
| Merge button | Disabled until marked ready | Enabled (subject to checks/reviews) |
| Reviewer auto-request | No (CODEOWNERS not pinged) | Yes |
| Review notifications | Quieter, most "please review" pings skip drafts | Pings reviewers immediately |
| Visual | Grey "Draft" badge | Green "Open" |
| `pull_request` workflow | Fires by default, CI **does** run on drafts unless filtered with `if: github.event.pull_request.draft == false` | Always runs |
| Branch protection gates | Exempt from "ready" gates (still subject to required reviews once converted) | Fully enforced |
| Convertible | "Mark as draft" ↔ "Ready for review" any time, no state lost | Same |

**Why use a draft:**
- Want CI feedback on a WIP branch without spamming reviewers
- Sharing an architectural sketch for early input
- Stacked PRs where the parent isn't ready yet

**CI gotcha.** A workflow triggered on `pull_request` to `main` with no draft filter runs on draft PRs too. And if the workflow's `push` trigger is scoped to `main` only, then pushing to a feature branch *without* opening a PR runs nothing at all. Opening the draft is what starts CI.
