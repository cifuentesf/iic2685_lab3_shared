from setuptools import setup
import os
from glob import glob

package_name = 'iic2685_lab3'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.*')),
        (os.path.join('share', package_name, 'mapas'), glob('mapas/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mcifuentesf',
    maintainer_email='mcifuentesf@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'likelihood_map.py = iic2685_lab3.likelihood_map:main',
            'particle_filter.py = iic2685_lab3.particle_filter:main',
            'motion_controller.py = iic2685_lab3.motion_controller:main',
        ],
    },
)