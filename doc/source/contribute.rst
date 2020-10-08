Contribute to EMD
=================

Thank you for your interest in contributing to EMD! Development of EMD takes place on `our gitlab page <https://gitlab.com/emd-dev/emd>`_. You can contribute to the developement of the EMD toolbox through gitlab by identifying and raising issues or by submitting new code through merge requests. Note that these will require an active account on `gitlab.com <https://www.gitlab.com>`_.

Both issues and merge requests are very welcome! We will try to resolve all issues but please bear in mind that this is a small open-source project with limited developement time.

Issues
------

Our `issue tracker <https://gitlab.com/emd-dev/emd/-/issues>`_ is the place to submit tickets about things that could or should change in the EMD toolbox. These could be bugs or about any problems you find, or requests for new functionality.

When submitting information about a bug, please include the following information so that the issue can be easily understood and reproduced.

- Expected Behaviour, what should happen when the code runs?
- Actual Behaviour, what is actually happening?
- Steps to Reproduce the Problem, what would another person need to do to see the issue?
- System Specifications, what operating system are you using? which versions of python and emd are you running?

Once the ticket has been submitted, we can take a look at the issue and may use the comments section at the bottom of the issue page to ask for more information and discuss a solution. Once a solution has been found, it will be submitted in a merge request linked to the issue in question.

Merge Requests
--------------

It it not possible to commit directly to the master branch of EMD, so all updates must first appear as `merge requests <https://gitlab.com/emd-dev/emd/-/merge_requests>`_. Each merge requests corresponds to a git branch containing a set of changes that may be added into EMD.

The merge request page provides a lot of useful information about the branch.

- Is the branch up to date with master? if not a rebase or merge may be required.
- Are the tests passing? We cannot merge code which breaks our tests! We test for a range of features including code behaviour, flake8 style compliance and spelling (using ``codespell``, this is particularly important for docstrings and tutorials)

At the start, all merge requests are marked as Work In Progress (WIP), this means that the code is not ready for merge yet. This tests can be run and process tracked as commits are added to the branch. Please feel free to use the comments section to discuss any changes and ask for feedback.

Once the changes are finished and the tests are passing you can click 'Resolve WIP status' to mark the branch as ready for merge. A developer can then review the changes before either a) accepting the request and merging or b) requesting some more information or changes using the comments section at the bottom of the page.
