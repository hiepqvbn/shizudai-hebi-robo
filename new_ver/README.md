# Shizuoka University Hebi-Robotics

A brief description of your project.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Contributing](#contributing)
- [License](#license)
- [Contact Me](#contact-me)

## Introduction

This is a research project conducted by the Kobayashi Lab at Shizuoka University, utilizing a Hebi-robot 5 DoF Arm. The objective of this research is to develop a grid learning algorithm-based controller capable of efficiently predicting the robot's movements within the visual range of an external camera and accounting for movement constraints. The aim is to optimize the robot's performance in scenarios where visualibility and movement constraints are crucial factors.

## Installation

Provide instructions on how to install and set up your project.

### Prerequisites

To run this project, the following prerequisites are required:

- Ubuntu 20 operating system
- Python version >= 3.8

Please make sure you have Ubuntu 20 installed on your machine. If not, you can download and install it from the [official Ubuntu website](https://ubuntu.com/download).

```bash
# Clone the repository
git clone https://github.com/hiepqvbn/shizudai-hebi-robo.git

# Checkout to new version branch
git checkout new_ver

# Change into the project directory
cd shizudai-hebi-robo/new_ver

# Activate the hebi-venv
source hebi-venv/bin/activate

# Or create new hebi-venv
python3 -m venv hebi-venv

# Install dependencies
pip install -r requirements.txt
```

## Modules

The project consists of the following modules:

- **simulation**: This module contains the `simulation_code.py` file that implements the simulation logic.

- **data_collection**: This module is responsible for collecting data both from the robot and the simulation environment. It contains the `data_collection.py` file, which provides functions and classes to gather and store data from various sources.

- **model_training**: This module contains the `model.py` file that defines the model architecture, and the `train.py` file that handles training the model.

- **robot_control**: This module is further divided into sub-modules:

  - **computer_vision**: This sub-module contains the `vision_code.py` file responsible for computer vision tasks.
  - **controller**: This sub-module contains the `controller_code.py` file that implements the robot controller logic.
  - **model_integration**: This sub-module contains the `integrate_model.py` file that integrates the trained model with the robot control system.

- **validation_testing**: This module is further divided into sub-modules:

  - **simulated_environment**: This sub-module contains the `test_simulation.py` file for testing the robot in a simulated environment.
  - **real_world_environment**: This sub-module contains the `test_real_world.py` file for testing the robot in a real-world environment.

- **documentation**: This module contains project documentation files such as `project_report.pdf` and `model_documentation.pdf`. It also has a sub-directory `code_documentation` where you can include your code documentation files.

- **data**: This module is further divided into sub-modules:

  - **simulation_data**: This sub-module stores the simulation data files.
  - **robot_data**: This sub-module stores the collected robot data files.

- **models**: This module stores the trained model file `trained_model.pth`.

In addition to these modules, the project also includes the following files:

- `requirements.txt`: This file lists the required dependencies for running the project.
- `README.md`: This file contains information about the project, its structure, and instructions for running it.

Feel free to modify and customize the project structure and modules according to your specific requirements.

## Contact Me

Feel free to contact me if you have any questions:

- **Facebook**: [Shizuoka University Hebi-Robotics](https://www.facebook.com/nxhiep)
- **LineID**: nxhiep97
- **Email**: nx.hiep97dz@gmail.com
