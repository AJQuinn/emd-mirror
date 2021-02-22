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

See below for detailed descriptions of the contribution process in a couple of different cases.

Create a merge-request....
**************************

.. container:: toggle body

    .. container:: header body

        .. raw:: html

            <h3 class='installbar'>... from an issue ticket</h3>

    .. container:: installbody body

        This section outlines how to contribute to by the issue-tracker on gitlab.com/emd-dev/emd. This is the best method to use if the changes will be made with contributions from several people.

        1. First, create an issue in the EMD `issue tracker <https://gitlab.com/emd-dev/emd/-/issues>`_. The issue should clearly introduce the potential changes that will be made.
        2. The issue will be read by an emd-dev developer who may ask for more information or suggest another solution in the issue discussion.
        3. If everyone agrees that a change is needed -  the developer will create a new branch and merge request which are specifically linked to the issue.
        4. The branch will be publicly accessible, you can install EMD with this branch using the methods in the `developer section of the install page <file:///Users/andrew/src/emd/doc/build/html/install.html#development-gitlab-com-version>`_.
        5. You and the developer can the work on the branch until the changes are finalised. Feel free to use the discussion section of the merge request.
        6. Once the work is complete, run some final checks on your local branch.
            - Ensure the tests are passing
            - Ensure that the code to be committed is flake8 compliant
            - Briefly describe the changes in the appropriate section of the changelog.
        7. The developer can then approve the merge request and the changes will be accepted into the main EMD branch.


.. container:: toggle body

    .. container:: header body

        .. raw:: html

            <h3 class='installbar'>... from a fork of EMD</h3>

    .. container:: installbody body

        This section outlines how to contribute to EMD from your own fork of the repository. This might be the simplest method if you would like to configure the gitlab.com environment and/or keep any changes private during development.

        1. First, create a fork of EMD from the `gitlab repository <https://gitlab.com/emd-dev/emd>`_.
        2. Create a branch for your changes in the fork of EMD. Any contributions must come from a branch - don't merge the branch into master in the forked repository.
        3. Ensure that runners are enabled so that tests can run on gitlab.com. "Settings -> CI/CD -> Public Pipelines" should be ticked in the gitlab.com settings.
        4. Complete your work on the branch.
        5. Run some final checks on your local branch.
            - Ensure that your branch is up to date with the main branch on emd-dev - this may require updating your fork.
            - Ensure the tests are passing
            - Ensure that the code to be committed is flake8 compliant
            - Briefly describe the changes in the appropriate section of the changelog.
        6. Submit the merge request from your fork of EMD. On your fork of gitlab.com go to "Repository -> Branches" and click 'Merge request" next to corresponding branch.
        7. The request will intially be marked as a Work In Progress (WIP). We will review the changes and potentially request some final changes or tweaks in the discussion on the Merge Request page.
        8. Once the developers are happy that the changes are ready, WIP status will be updated and the branch merged into the main EMD branch.


