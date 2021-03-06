# Changelog

Changes and updates to EMD are tracked by version on this page.  The format of
this changelog is (mostly) based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this the EMD package uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changes should be categorised under the following types:

- **Added** for new features.
- **Changed** for changes in existing functionality.
- **Deprecated** for soon-to-be removed features.
- **Removed** for now removed features.
- **Fixed** for any bug fixes.
- **Security** in case of vulnerabilities.

Where appropriate, links to specific Issues & Merge Requests on [our gitlab page](https://gitlab.com/emd-dev/emd).

---

## Development Version
Work in progress...

    git clone https://gitlab.com/emd-dev/emd.git

### Added
- New function for computing summary stats from chains of cycles (from marcoFabus) [!46](https://gitlab.com/emd-dev/emd/-/merge_requests/46)

### Changed
- Major updates to tutorials
  - Binder notebooks added
  - New sifting tutorials added

---

# Stable Versions

## 0.3.2

    pip install emd==0.3.2
Released 2020-11-29

### Added
- Add input array shape ensurance functions and start to use in sift & cycle submodules  [!26](https://gitlab.com/emd-dev/emd/-/merge_requests/26)
- Add more stopping criteria to sift module [!27](https://gitlab.com/emd-dev/emd/-/merge_requests/26)
  - Rilling et al and fixed iterations IMF stopping criteria
  - Energy threshold sift stopping criterion


### Changed
- Refactor some options extrema detection functions [!29](https://gitlab.com/emd-dev/emd/-/merge_requests/29)
- Sift throws an error if an IMF doesn't converge after a specified maximum number of iterations.
- Refactor mask generation in mask sift. Now specifies N masks of different phases and has options for parallel processing.
- SiftConfig yaml file also stores which type of sift the config is for [!35](https://gitlab.com/emd-dev/emd/-/merge_requests/35)
- 18% increase in testing coverage (to 75% total) [!30](https://gitlab.com/emd-dev/emd/-/merge_requests/30)

### Deprecated
- emd.spectra.frequency_stats renamed to emd.spectra.frequency_transform. Original func kept for now.

---

## 0.3.1

    pip install emd==0.3.1
Released 2020-09-06

### Added
- This changelog [!18](https://gitlab.com/emd-dev/emd/-/merge_requests/18)
- support.py submodule with some helper functions for checking installs and running tests [!20](https://gitlab.com/emd-dev/emd/-/merge_requests/20)
- envs subdir containing anaconda install environment config files [!21](https://gitlab.com/emd-dev/emd/-/merge_requests/21)
- Options for reading and writing sift configs to yaml format [!24](https://gitlab.com/emd-dev/emd/-/merge_requests/24)
- major update to webpage [!12](https://gitlab.com/emd-dev/emd/-/merge_requests/24)
  - Reformat page to bootstrap
  - Add & update the tutorials
  - New landing page

### Fixed
- Input array dimensions in phase_align clarified and fixed up [ef28b36c](https://gitlab.com/emd-dev/emd/-/commit/ef28b36cac8be7224280fd7ba02d25b3f084ab30)
- Extrema opts were dropped in get_next_imf [!23](https://gitlab.com/emd-dev/emd/-/merge_requests/23)

### Changed
- get_control_points internal refector [af153ed6](https://gitlab.com/emd-dev/emd/-/commit/af153ed606601f3963c125329c86710e47c06b45)

---

## 0.3.0

    pip install emd==0.3.0
Released on 2020-07-22

### Added
- get_cycle_stat refectored to allow general numpy and user-specified metrics to be computed
- Logger coverage increased, particularly in cycle.py
  - Logger exit message added

### Changed
- Major SiftConfig refactor - API & syntax now much cleaner

---

## 0.2.0

    pip install emd==0.2.0
Released 2020-06-05

### Added
- Tutorials on the sift, hilbert-huang and holospectrum analyses.
- Parabolic extrema interpolation
- Average envelope scaling in sift
- Testing expanded to include python 3.5, 3.6, 3.7 & 3.8


### Changed
- API in sift functions updated for compatabillity with new SiftConfig
  - Expose options for extrema padding to top level sift function
  - Sift relevant util functions moved into sift.py submodule
- Masked sift functions merged into single function
- get_cycle_chain refactor to cleaner internal syntax

---

## 0.1.0

    pip install emd==0.1.0
Released 2019-12-10

### Added
- Everything
