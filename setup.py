from distutils.core import setup

setup(
    name = 'viabel',
    version='0.1',
    description="Efficient, lightweight, variational inference approximation bounds",
    author='Jonathan H. Huggins',
    author_email='jhuggins@mit.edu',
    url='https://github.com/jhhuggins/viabel/',
    packages=['viabel'],
    install_requires=['numpy'],
    python_requires='>=2.7',
    keywords = ['Bayesian inference',
                'variational inference',
                'Wasserstein distance'],
    platforms='ALL',
)
