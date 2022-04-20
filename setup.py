from setuptools import setup, find_packages

setup(
        name = 'project 2',
        version = '1.0',
        author = 'Kaustubh Pande',
        author_email = 'kaustubhpande@ou.edu',
        packages = find_packages(exclude = ('tests', 'docs')),
        setup_requires = ['pytest-runner'],
        tests_require = ['pytest']
    )
