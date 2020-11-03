=========================
Installation and Testing
=========================

To get the latest released version, just use pip::

    $ pip install viabel

However, ``VIABEL`` is under active development, and you may want to install
the latest version from github::

    $ pip install git+git://github.com/jhuggins/viabel

To run the tests, in the root of the repository, run::

    $ python3 -m pytest

To see code coverage, in the root of the repository, run::

    $ coverage run --include='viabel/[A-Za-z]*.py' -m pytest
    $ coverage html

Then view the ``htmlcov/index.html`` in your web browser.
