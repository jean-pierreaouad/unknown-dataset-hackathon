# Automated verification

The first cross-platform run passed on 22 October 2026:
[Python tests — run 39823152794](https://github.com/jean-pierreaouad/unknown-dataset-hackathon/actions/runs/39823152794).

Tested code commit: `1ab0d26f240fbde1c22d95fadc7121f6ce8da031`.
Later presentation and documentation commits retain the same Python code,
dependencies, example data, and workflow.

| GitHub-hosted runner | Python | Dependency installation and pip check | Tests | Included examples |
|---|---|---|---|---|
| Windows | 3.12 | Passed | 47 passed; 28.439 seconds for the suite | All three passed their output checks |
| Ubuntu Linux | 3.12 | Passed | 47 passed; 23.222 seconds for the suite | All three passed their output checks |

These suite times are not per-dataset timing measurements. The workflow creates
fresh Python environments and installs `requirements.txt`, then runs the
standard unittest command and each example through the real entry point and
output checker. Job logs confirm the 47-test count and three checker successes.

The workflow runs when code or workflow files change on `main`, on pull
requests, and on manual dispatch. Documentation-only pushes are skipped.
It requests read-only repository contents permission and uses the official
[checkout](https://github.com/actions/checkout) and
[setup-python](https://github.com/actions/setup-python) actions.

The README badge links to the latest workflow status. This recorded run is
historical evidence for the specified code. GitHub runners are not the
organizer's assessment image; this does not certify the official interface,
private-dataset quality, a contest score, or a course submission.
