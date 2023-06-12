from setuptools import setup
import os
from glob import glob
package_name = 'ts_controllers'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['resource/path.csv']),
        (os.path.join('share', package_name), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lukasz',
    maintainer_email='lukasz@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_ctrl = ts_controllers.mpc_ctrl:main',
            'stanley_ctrl = ts_controllers.stanley:main',
            'pure_pursuit_ctrl = ts_controllers.pure_pursuit:main',
            'path_publisher = ts_controllers.path:main',
        ],
    },
)
