from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name = 'viabel',
    version='0.3.0.post1',
    description='Efficient, lightweight, variational inference and approximation bounds',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jonathan H. Huggins',
    author_email='huggins@bu.edu',
    url='https://github.com/jhuggins/viabel/',
    packages=['viabel'],
    install_requires=['numpy>=1.13'],
    extras_require={'vb' : ['scipy', 'autograd', 'tqdm', 'paragami'],
                    'examples' : ['scipy', 'autograd', 'tqdm', 'paragami',
                                  'matplotlib', 'seaborn', 'pandas', 'pystan']},
    python_requires='>=2.7',
    keywords = ['Bayesian inference',
                'variational inference',
                'Wasserstein distance',
                'alpha-divergence'],
    classifiers=['Development Status :: 4 - Beta',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.5'],
    platforms='ALL',
)
