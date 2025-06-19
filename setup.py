from setuptools import setup

package_name = 'iic2685_lab3'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mcifuentesf',
    maintainer_email='tu_correo@ejemplo.cl',
    description='Paquete para laboratorio 3 de IIC2685',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_publisher = iic2685_lab3.simple_publisher:main',
        ],
    },
)
