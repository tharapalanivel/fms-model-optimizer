<!-- Thank you for the contribution! -->

### Description of the change

<!-- Please summarize the changes -->

### Related issues or PRs

<!-- For example: "Closes #1234" or "Fixes bug introduced in #5678 -->

### (Optional) List any documentation or testing added

<!-- Describe which features were documented/tested -->

### (Optional) How to verify the contribution

<!-- Provide instructions on how to verify your contribution if unit tests do not provide coverage -->

### Checklist for passing CI/CD:

<!-- Mark completed tasks with "- [x]" -->
- [ ] All commits are signed showing "Signed-off-by: Name \<email@domain.com\>" with `git commit -signoff` or equivalent
- [ ] PR title and commit messages adhere to [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
- [ ] Contribution is formatted with `tox -e fix`
- [ ] Contribution passes linting with `tox -e lint`
- [ ] Contribution passes spellcheck with `tox -e spellcheck`
- [ ] Contribution passes all unit tests with `tox -e unit`

Note: CI/CD performs unit tests on multiple versions of Python from a fresh install.  There may be differences with your local environment and the test environment.