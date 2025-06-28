from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'iic2685_lab3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
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
            'sensor_model = iic2685_lab3.sensor_model:main',
            'particle_filter = iic2685_lab3.particle_filter:main',
            'visualize_sensor_model = iic2685_lab3.visualize_sensor_model:main',
        ],
    },
)
