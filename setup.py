try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

exec(open('bmcmc/version.py').read())

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='bmcmc',
    version=__version__,
    description='An MCMC package for Bayesian data analysis',
    long_description=long_description,
    url='https://github.com/sanjibs/bmcmc',
    author='Sanjib Sharma',
    author_email='bug.sanjib@gmail.com',
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
    ],
    install_requires=['ebfpy'],
    packages=['bmcmc'],
    include_package_data=True,
    package_data={'': ['AUTHORS.rst','README.rst','LICENSE']},
)

