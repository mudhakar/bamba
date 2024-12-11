# Contributing Guidelines

### Contents

- [Code of Conduct](#book-code-of-conduct)
- [Opening an Issue](#inbox_tray-opening-an-issue)
- [Submitting Pull Requests](#repeat-submitting-pull-requests)
- [Code Review](#white_check_mark-code-review)

> **This guide serves to set clear expectations for everyone involved with the project so that we can improve it together while also creating a welcoming space for everyone to participate. Following these guidelines will help ensure a positive experience for contributors and maintainers.**

## :book: Code of Conduct

Please review our [Code of Conduct](code-of-conduct.md). It is in effect at all times. We expect it to be honored by everyone who contributes to this project. 

## :inbox_tray: Opening an Issue

### :beetle: Bug Reports

A great way to contribute to the project is to send a detailed issue when you encounter a problem using our models. We always appreciate a well-written, thorough bug report. :v:

Please open an issue with the [Bug Report template]() and fill out all the required information.

- **Do not open a duplicate issue!** Search through existing issues to see if your issue has previously been reported. If your issue exists, comment with any additional information you have.

- **Fully complete the provided issue template.** The bug report template requests all the information we need to quickly and efficiently address your issue. Be clear, concise, and descriptive. Provide as much information as you can, including steps to reproduce, stack traces, compiler errors, library versions, OS versions, and screenshots (if applicable).

### :love_letter: Feature Requests

Feature requests are welcome! We will consider all requests, and get back on whether we are able to accept the request and if it fits in with the vision of the project. We may ask you to create a pull request and contribute the desired feature in some cases. 

To create a feature request, open an issue with [Feature Request]() template

- **Do not open a duplicate feature request.** Search for existing feature requests first. If you find your feature (or one very similar) previously requested, comment on that issue.

- **Fully complete the provided issue template.** The feature request template asks for all necessary information for us to begin a productive conversation. 

- Be precise about the proposed outcome of the feature and how it relates to existing features. Include implementation details if possible.

## :repeat: Submitting Pull Requests

We **love** pull requests! Before [forking the repo](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) and [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests) for non-trivial changes, it is usually best to first open an issue to discuss the changes, or discuss your intended approach for solving the problem in the comments for an existing issue.

- **Smaller is better.** Submit **one** pull request per bug fix or feature. A pull request should contain isolated changes pertaining to a single bug fix or feature implementation. **Do not** refactor or reformat code that is unrelated to your change. It is better to **submit many small pull requests** rather than a single large one. Enormous pull requests will take enormous amounts of time to review, or may be rejected altogether. 

- **Coordinate bigger changes.** For large and non-trivial changes, open an issue to discuss a strategy with the maintainers.

- **Prioritize understanding over cleverness.** Write code clearly and concisely. Remember that source code usually gets written once and read often. Ensure the code is clear to the reader. The purpose and logic should be obvious to a reasonably skilled developer, otherwise you should add a comment that explains it.

- **Follow existing coding style and conventions.** Keep your code consistent with the style, formatting, and conventions in the rest of the code base. When possible, these will be enforced with a linter. Consistency makes it easier to review and modify in the future.

- **Include test coverage.** Add unit tests or UI tests when possible. Follow existing patterns for implementing tests.

- **Update the example project** if one exists to exercise any new functionality you have added.

- **Add documentation.** Document your changes with code doc comments or in existing guides.

- **Use the repo's default branch.** Branch from and [submit your pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork) to the repo's default branch. Usually this is `main`, but it could be `dev`, `develop`, or `master`.

- **[Resolve any merge conflicts](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-on-github)** that occur.

- **Promptly address any CI failures**. If your pull request fails to build or pass tests, please push another commit to fix it. 

- When writing comments, use properly constructed sentences, including punctuation.

- Use spaces, not tabs.

- **Write detailed commit messages**. Commit messages should explain what is being solved.


## :white_check_mark: Code Review

Once you've created a pull request, maintainers will review your code and may make suggestions to fix before merging. It will be easier for your pull request to receive reviews if you consider the criteria the reviewers follow while working. Remember to:

    Run tests locally and ensure they pass
    Follow the project coding conventions
    Write detailed commit messages
    Break large changes into a logical series of smaller patches, which are easy to understand individually and combine to solve a broader issue

Maintainers are encouraged to perform "squash and merge" actions on PRs in this repository. Thus, it doesn't matter how many commits your PR has, as they will end up being a single commit after merging.

