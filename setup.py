from distutils.core import setup

setup(
    name='HMMreps',
    version='0.0.1',
    author='S. Suster',
    author_email='sim.suster@gmail.com',
    packages=['hmm'],
    url='https://github.com/SimonSuster',
    license='LICENSE.txt',
    description='Hidden Markov models for word representations',
    keywords="",
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.9",
        "sselogsumexp"
    ],
)
