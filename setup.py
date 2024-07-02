from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

class CMakeBuildExt(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
        ]
        
        build_args = []

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        os.chdir(self.build_temp)
        self.spawn(['cmake', os.path.abspath(ext.sourcedir)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        os.chdir(os.path.abspath(os.path.dirname(__file__)))

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='qcos',
    version='0.1.0',
    author='Govind Chari',
    author_email='govindchari1@gmail.com',
    description='QCOS: Quadratic Objective Conic Optimization Solver',
    long_description=open('README.md').read(),
    install_requires=required,
    ext_modules=[CMakeExtension('qcos', sourcedir='src')],
    cmdclass={'build_ext': CMakeBuildExt},
    package_dir={'': 'src'},
    packages=['qcos'],
    zip_safe=False,
)
