# Copyright 2021 Machine Learning Research @ Bayer AG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install script for setuptools."""

from setuptools import setup


setup(
    name='img2mol',
    version='0.1',
    packages=['img2mol'],
    url='https://github.com/bayer-science-for-a-better-life/Img2Mol',
    license='Apache License, Version 2.0',
    author='Djork-Arné Clevert, Tuan Le, Robin Winter and Floriane Montanari',
    author_email='djork-arne.clevert@bayer.com',
    description='Inferring molecules from images'
)
