from os import path
from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()


setup(
    name = 'viabel',
    version='0.3.0.post1',
    description='Efficient, lightweight variational inference and approximation bounds',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Jonathan H. Huggins',
    author_email='huggins@bu.edu',
    url='https://github.com/jhuggins/viabel/',
    packages=['viabel'],
    install_requires=['numpy>=1.13', 'scipy', 'autograd', 'tqdm', 'paragami'],
    extras_require={'examples' : ['matplotlib', 'seaborn', 'pandas', 'pystan', 'ipython', 'jupyter']},
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
