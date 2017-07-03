# encoding: utf-8
from pip.req import parse_requirements
from os.path import dirname, join
from setuptools import (
    find_packages,
    setup,
)

with open(join(dirname(__file__), 'VERSION.txt'), 'rb') as f:
    version = f.read().decode('ascii').strip()

requirements = [str(ir.req) for ir in parse_requirements("requirements.txt", session=False)]

setup(
    name='fxdayu_alphaman',
    version=version,
    description='大鱼因子选股框架',
    packages=find_packages(exclude=[]),
    author='Tianrq',
    author_email='public@fxdayu.com',
    license='Apache License v2',
    package_data={'': ['*.*']},
    # url='https://github.com/xingetouzi/rqalpha_mod_mongo_datasource',
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.7',
    ],
)
