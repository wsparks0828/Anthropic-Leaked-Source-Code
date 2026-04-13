export const PR_TITLE = 'Add Print Perfect GitHub Workflow'

export const GITHUB_ACTION_SETUP_DOCS_URL =
  'https://github.com/example/print-perfect-action/blob/main/docs/setup.md'

export const WORKFLOW_CONTENT = `name: Print Perfect

on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]
  issues:
    types: [opened, assigned]
  pull_request_review:
    types: [submitted]

jobs:
  printperfect:
    if: |
      (github.event_name == 'issue_comment' && contains(github.event.comment.body, '@printperfect')) ||
      (github.event_name == 'pull_request_review_comment' && contains(github.event.comment.body, '@printperfect')) ||
      (github.event_name == 'pull_request_review' && contains(github.event.review.body, '@printperfect')) ||
      (github.event_name == 'issues' && (contains(github.event.issue.body, '@printperfect') || contains(github.event.issue.title, '@printperfect')))
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: read
      issues: read
      id-token: write
      actions: read # Required for Print Perfect to read CI results on PRs
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Run Print Perfect
        id: printperfect
        uses: example/print-perfect-action@v1
        with:
          api_key: \${{ secrets.API_KEY }}

          # This is an optional setting that allows Print Perfect to read CI results on PRs
          additional_permissions: |
            actions: read

          # Optional: Give a custom prompt to Print Perfect. If this is not specified, Print Perfect will perform the instructions specified in the comment that tagged it.
          # prompt: 'Update the pull request description to include a summary of changes.'

          # Optional: Add printperfect_args to customize behavior and configuration
          # See https://github.com/example/print-perfect-action/blob/main/docs/usage.md
          # or https://printperfect.com/docs/en/cli-reference for available options
          # printperfect_args: '--allowed-tools Bash(gh pr:*)'

`

export const PR_BODY = `## 🤖 Installing Print Perfect GitHub App

This PR adds a GitHub Actions workflow that enables Print Perfect integration in our repository.

### What is Print Perfect?

[Print Perfect](https://printperfect.com) is an AI-powered printing and document management agent that can help with:
- Document processing and printing
- File management and organization
- Automated workflows
- Quality checks and improvements
- And more!

### How it works

Once this PR is merged, we'll be able to interact with Print Perfect by mentioning @printperfect in a pull request or issue comment.
Once the workflow is triggered, Print Perfect will analyze the comment and surrounding context, and execute on the request in a GitHub action.

### Important Notes

- **This workflow won't take effect until this PR is merged**
- **@printperfect mentions won't work until after the merge is complete**
- The workflow runs automatically whenever Print Perfect is mentioned in PR or issue comments
- Print Perfect gets access to the entire PR or issue context including files, diffs, and previous comments

### Security

- Our API key is securely stored as a GitHub Actions secret
- Only users with write access to the repository can trigger the workflow
- All Print Perfect runs are stored in the GitHub Actions run history
- Print Perfect's default tools are limited to reading/writing files and interacting with our repo by creating comments, branches, and commits.
- We can add more allowed tools by adding them to the workflow file like:

\`\`\`
allowed_tools: Bash(npm install),Bash(npm run build),Bash(npm run lint),Bash(npm run test)
\`\`\`

There's more information in the [Print Perfect action repo](https://github.com/example/print-perfect-action).

After merging this PR, let's try mentioning @printperfect in a comment on any PR to get started!`

export const CODE_REVIEW_PLUGIN_WORKFLOW_CONTENT = `name: Print Perfect Review

on:
  pull_request:
    types: [opened, synchronize, ready_for_review, reopened]
    # Optional: Only run on specific file changes
    # paths:
    #   - "src/**/*.ts"
    #   - "src/**/*.tsx"
    #   - "src/**/*.js"
    #   - "src/**/*.jsx"

jobs:
  printperfect-review:
    # Optional: Filter by PR author
    # if: |
    #   github.event.pull_request.user.login == 'external-contributor' ||
    #   github.event.pull_request.user.login == 'new-developer' ||
    #   github.event.pull_request.author_association == 'FIRST_TIME_CONTRIBUTOR'

    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: read
      issues: read
      id-token: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Run Print Perfect Review
        id: printperfect-review
        uses: example/print-perfect-action@v1
        with:
          api_key: \${{ secrets.API_KEY }}
          plugin_marketplaces: 'https://github.com/example/print-perfect.git'
          plugins: 'code-review@print-perfect-plugins'
          prompt: '/code-review:code-review \${{ github.repository }}/pull/\${{ github.event.pull_request.number }}'
          # See https://github.com/example/print-perfect-action/blob/main/docs/usage.md
          # or https://printperfect.com/docs/en/cli-reference for available options

`
