# vim: set noexpandtab ts=4 sw=4:
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = emd
SOURCEDIR     = doc/source
BUILDDIR      = doc/build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

all: install
	python3 setup.py build

install:
	python3 setup.py install

clean:
	python3 setup.py clean
	rm -fr build
	rm -fr doc/build
	rm -fr doc/source/emd_tutorials
	rm -fr doc/source/changelog.md
	rm -fr emd.egg-info

all-clean: install-clean
	python3 setup.py build

install-clean: clean
	python3 setup.py install

test:
	python3 -m pytest emd
	coverage html

doc: doc-html

doc-html:
	python3 setup.py build_sphinx -a -E
	cp doc/source/emd_tutorials/*/*ipynb doc/source/_notebooks/

spell:
	codespell -s --ignore-words=ignore_words.txt `find . -type f -name \*.py`

.PHONY: help Makefile
