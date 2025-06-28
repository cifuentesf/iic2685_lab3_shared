from setuptools import find_packages, setup

package_name = 'iic2685_lab3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/localization.xml']),
        ('share/' + package_name + '/maps', ['maps/mapa.pgm', 'maps/mapa.yaml']),
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
            'sensor_model.py = iic2685_lab3.sensor_model:main',
            'particle_filter.py = iic2685_lab3.particle_filter:main',
            'exploration_navigator.py = iic2685_lab3.exploration_navigator:main',
            'localization_manager.py = iic2685_lab3.localization_manager:main',
        ],
    },
)