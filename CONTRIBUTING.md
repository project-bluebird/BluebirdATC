# Contributing to BluebirdATC

 **Welcome to BluebirdATC!**
 We're happy that you want to contribute!

 These guidelines are intended to make it as easy as possible to get involved.
 We welcome all contributions, whether it is
 * reporting bugs
 * requesting new features
 * adding, improving, or fixing documentation
 * suggesting, writing or improving tests
 * writing code
 * anything else!

 Don't worry about making everything perfect - it's more important to start the conversation than to have everything right first time.

 ## Table of contents

 - [Where to start: issues](#where-to-start-issues)
 - [Making a change with a pull request](#making-a-change-with-a-pull-request)
   - [1. Comment on an existing issue or open a new issue referencing your addition](#1-comment-on-an-existing-issue-or-open-a-new-issue-referencing-your-addition)
   - [2. Fork the BluebirdATC repository to your profile](#2-fork-the-bluebird-atc-repository-to-your-profile)
   - [3. Make the changes you've discussed](#3-make-the-changes-youve-discussed)
   - [4. Submit a pull request](#4-submit-a-pull-request)
 - [Running the tests](#running-the-tests)
 - [Style guide](#style-guide)

 ## Where to start: issues

 * **Issues** are individual pieces of work that need to be completed to move the project forwards.
 A general guideline: if you find yourself tempted to write a great big issue that
 is difficult to describe as one unit of work, please consider splitting it into two or more subissues.

 Before you open a new issue, please check if any of our [open issues](https://github.com/project-bluebird/BluebirdATC/issues) covers your idea already.


 ## Making a change with a pull request

 We appreciate all contributions to `BluebirdATC`.
 **THANK YOU** for helping us.

 All project management, conversations and questions related to the `BluebirdATC` project happens here in the [BluebirdATC repository][bluebird-atc-repo].

 The following steps are a guide to help you contribute in a way that will be easy for everyone to review and accept with ease.

 ### 1. Comment on an [existing issue](https://github.com/project-bluebird/BluebirdATC/issues) or open a new issue referencing your addition

 This allows other members of the `BluebirdATC` team to confirm that you aren't overlapping with work that's currently underway and that everyone is on the same page with the goal of the work you're going to carry out.
 
 As this digital twin is heavily domain driven for ATC, issues are also a perfect point of discussion to gather additional context about the changes you are proposing, and if they align with the domain.

 [This blog](https://www.igvita.com/2011/12/19/dont-push-your-pull-requests/) is a nice explanation of why putting this work in up front is so useful to everyone involved.

 ### 2. [Fork][github-fork] the [BluebirdATC repository][bluebird-atc-repo] to your profile

 This is now your own unique copy of `BluebirdATC`.
 Changes here won't affect anyone else's work, so it's a safe space to explore edits to the code!

 Make sure to [keep your fork up to date][github-syncfork] with the master repository, otherwise you can end up with lots of dreaded [merge conflicts][github-mergeconflicts].

 ### 3. Make the changes you've discussed

 Try to keep the changes focused.
 If you submit a large amount of work all in one go it will be much more work for whomever is reviewing your pull request.

 While making your changes, commit often and write good, detailed commit messages.
 [This blog](https://chris.beams.io/posts/git-commit/) explains how to write a good Git commit message and why it matters.
 It is also perfectly fine to have a lot of commits - including ones that break code.
 A good rule of thumb is to push up to GitHub when you _do_ have passing tests then the continuous integration (CI) has a good chance of passing everything.

 If you feel tempted to "branch out" then please make a [new branch][github-branches] and a [new issue][bluebird-atc-issues] to go with it. [This blog](https://nvie.com/posts/a-successful-git-branching-model/) details the different Git branching models.

 Please do not re-write history!
 That is, please do not use the [rebase](https://help.github.com/en/articles/about-git-rebase) command to edit previous commit messages, combine multiple commits into one, or delete or revert commits that are no longer necessary.

 ### 4. Submit a [pull request][github-pullrequest]

 We encourage you to open a pull request as early in your contributing process as possible.
 This allows everyone to see what is currently being worked on.
 It also provides you, the contributor, feedback in real time from both the community and the continuous integration as you make commits (which will help prevent stuff from breaking).

 When you are ready to submit a pull request, make sure the contents of the pull request body do the following:
 - Describe the problem you're trying to fix in the pull request, reference any related issues and use keywords fixes/close to automatically close them, if pertinent.
 - List changes proposed in the pull request.
 - Describe what the reviewer should concentrate their feedback on.

 If you have opened the pull request early and know that its contents are not ready for review or to be merged, mark it as a draft pull request".
 When you are happy with it and are happy for it to be merged into the main repository, change status to Ready for review".

 A member of the BluebirdATC team will then review your changes to confirm that they can be merged into the main repository.
 A [review][github-review] will probably consist of a few questions to help clarify the work you've done.
 Keep an eye on your GitHub notifications and be prepared to join in that conversation.

 You can update your [fork][github-fork] of the [BluebirdATC repository][bluebird-atc-repo] and the pull request will automatically update with those changes.
 You don't need to submit a new pull request when you make a change in response to a review.

 You can also submit pull requests to other contributors' branches!
 Do you see an [open pull request](https://github.com/project-bluebird/BluebirdATC/pulls) that you find interesting and want to contribute to?
 Simply make your edits on their files and open a pull request to their branch!

 What happens if the continuous integration (CI) fails (for example, if the pull request notifies you that "Some checks were not successful")?
 The CI could fail for a number of reasons.
 At the bottom of the pull request, where it says whether your build passed or failed, you can click “Details” next to the test, which takes you to the Github Actions page.
 You can view the log or rerun the checks if you have write access to the repo by clicking the “Restart build” button in the top right

 GitHub has a [nice introduction][github-flow] to the pull request workflow, but please get in touch if you have any questions.

 ## Use of AI coding assistants

We recognise that the use of AI tools for writing code is becoming more and more widespread, and that these tools are becoming increasingly powerful. However, we request that there be a "human in the loop" for any contributions to BluebirdATC. In particular:

 - We will reject any pull request that we judge to have been made by an autonomous AI agent.
 - If you have used AI tools to help write code in a pull request, we ask that you (the human) have a good understanding of what has been changed and why.
 - Please let us know in the pull request text if you have made use of AI tools, and roughly to what extent - this will help us to get a picture of where the code is coming from.

 ## Running the tests

 The project uses `pytest` and relies on `uv` to manage the virtual environment.
 Assuming you have installed the dev dependencies (via `uv sync`), you can run the whole
 test suite from the repository root with:

 ```bash
 uv run pytest -n auto
 ```

 The `-n auto` flag runs the tests in parallel across your CPU cores (via `pytest-xdist`).
 This is the same command that continuous integration (CI) runs, so anything that passes
 locally is expected to pass in CI too.

 To run just one package's tests, pass its test directory explicitly. For example, to run
 only the `bluebird-dt` tests:

 ```bash
 uv run pytest bluebird-dt/tests
 ```

 The same works for any of the other test suites, e.g. `bluebird-api/tests`,
 `bluebird-gymnasium/tests` and `docs/tests`. See the `testpaths` setting in the root
 `pyproject.toml` for the full list.

 ## Style Guide

 The python code itself should follow [PEP8][link_pep8] convention whenever possible.

 ---

 _These Contributing Guidelines have been adapted from [The Turing Way Contributing Guidelines](https://github.com/alan-turing-institute/the-turing-way/blob/master/CONTRIBUTING.md)! (License: MIT)_

 [bluebird-atc-repo]: https://github.com/project-bluebird/BluebirdATC
 [bluebird-atc-issues]: https://github.com/project-bluebird/BluebirdATC/issues
 [git]: https://git-scm.com
 [github]: https://github.com
 [github-branches]: https://help.github.com/articles/creating-and-deleting-branches-within-your-repository
 [github-fork]: https://help.github.com/articles/fork-a-repo
 [github-flow]: https://guides.github.com/introduction/flow
 [github-mergeconflicts]: https://help.github.com/articles/about-merge-conflicts
 [github-pullrequest]: https://help.github.com/articles/creating-a-pull-request
 [github-review]: https://help.github.com/articles/about-pull-request-reviews
 [github-syncfork]: https://help.github.com/articles/syncing-a-fork
 [link_pep8]: https://www.python.org/dev/peps/pep-0008/
