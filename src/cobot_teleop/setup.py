from setuptools import find_packages, setup

package_name = 'cobot_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='luefit00@hs-esslingen.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pedal_node = cobot_teleop.pedal_node:main',
            'replay_node = cobot_teleop.replay_node:main',
            'hw_test = cobot_teleop.hw_test:main',
        ],
    },
)
