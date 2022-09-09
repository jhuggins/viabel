import codecs
import os
from setuptools import setup

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")
REQUIREMENTS_DOCS_FILE = os.path.join(PROJECT_ROOT, "requirements-docs.txt")
REQUIREMENTS_DEV_FILE = os.path.join(PROJECT_ROOT, "requirements-dev.txt")
README_FILE = os.path.join(PROJECT_ROOT, "README.md")


def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()


def get_requirements_dev():
    with codecs.open(REQUIREMENTS_DEV_FILE) as buff:
        return buff.read().splitlines()


def get_requirements_docs():
    with codecs.open(REQUIREMENTS_DOCS_FILE) as buff:
        return buff.read().splitlines()


def get_long_description():
    with codecs.open(README_FILE, "rt") as buff:
        return buff.read()


setup(
    name = 'viabel',
    version='0.5.1',
    description='Efficient, lightweight variational inference and approximation bounds',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Jonathan H. Huggins',
    author_email='huggins@bu.edu',
    url='https://github.com/jhuggins/viabel/',
    packages=['viabel'],
    include_package_data=True,
    install_requires=get_requirements(),
    extras_require={ 'docs' : get_requirements_docs(),
                     'dev' : get_requirements_dev()
                     },
    python_requires='>=3.5',
    classifiers=['Programming Language :: Python :: 3',
                 'Natural Language :: English',
                 'License :: OSI Approved :: MIT License',
                 'Intended Audience :: Science/Research',
                 'Development Status :: 2 - Pre-Alpha',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering :: Mathematics'],
    platforms='ALL',
)
