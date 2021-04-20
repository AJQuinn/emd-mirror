import os


def find_bib():
    """Bibliography might be in slightly different places depending on build
    location."""

    if os.path.isfile('references.bib'):
        # Take local file
        refpath = 'references.bib'
    elif os.path.isfile(os.path.join(os.getcwd(), 'references.bib')):
        refpath = os.path.join(os.getcwd(), 'references.bib')
    elif os.path.isfile(os.path.join(os.getcwd(), 'doc', 'source', 'references.bib')):
        refpath = os.path.join(os.getcwd(), 'doc', 'source', 'references.bib')
    else:
        raise ValueError('Bibligraphy not found!')

    return refpath


def find_cite_template():
    """Bibliography might be in slightly different places depending on build
    location."""

    if os.path.isfile('cite_template.rst'):
        # Take local file
        refpath = 'cite_template.rst'

    elif os.path.isfile(os.path.join(os.getcwd(), 'cite_template.rst')):
        refpath = os.path.join(os.getcwd(), 'cite_template.rst')

    elif os.path.isfile(os.path.join(os.getcwd(), 'doc', 'source', 'cite_template.rst')):
        refpath = os.path.join(os.getcwd(), 'doc', 'source', 'cite_template.rst')

    else:
        raise ValueError('cite_template.rst not found!')

    return refpath


def load_references():
    refpath = find_bib()
    print('Loading bib file: {0}'.format(refpath))
    with open(refpath, 'r') as f:
        lib = f.readlines()

    refs = {}
    ref = None
    for idx, line in enumerate(lib):
        if (ref is None) and (line[0] == '@'):
            if ref is None:
                key = line.split('{')[1][:-2]
                print('Loading : {0}'.format(key))
                ref = {}
        else:
            if len(line) < 3 and line.find('}') > -1:
                # We're finished
                refs[key] = ref
                ref = None
            elif len(line) > 3:
                k, v = line.split('=')
                v = v.lstrip(' {').rstrip(' },\n')
                k = k.lstrip(' {').rstrip(' },\n')
                if k == 'author':
                    v = ', '.join(v.split(' and '))
                    start, _, end = v.rpartition(',')
                    v = start + ' &' + end
                ref[k] = v
    return refs


def build_citations():
    refs = load_references()

    temp = """
    <div class="container" style="margin-bottom:10px; padding-left: 35px; text-indent: -40px">
      {author} ({year})<br>
      <strong><font size="3px">{title}</font></strong><br>
      {journal} <a href=https://www.doi.org/{doi}>{doi}</a>
    </div>
"""

    # Read in the file
    temppath = find_cite_template()
    with open(temppath, 'r') as f:
        filedata = f.read()

    # Replace the target string
    for key in refs.keys():
        r = refs[key]
        cite = temp.format(**r)
        filedata = filedata.replace("{{{0}}}".format(key), cite)

    # Write the file out again
    citepath = temppath.replace('cite_template.rst', 'cite.rst')
    with open(citepath, 'w') as f:
        f.write(filedata)
