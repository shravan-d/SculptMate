# SculptMate

## Overview

SculptMate is a Blender add-on that simplifies the process of generating human meshes. It automates a significant part of the character creation pipeline, allowing users to quickly create human meshes based on a single image.

## Features

- Generate human meshes from single images.
- Automate 30% of the character generation pipeline.

## Demo

![Samples](assets/samples.gif)


## Installation

1. Download the latest release zip file from the [Releases](https://github.com/shravan-d/SculptMate/releases) page.
2. In Blender, go to `Edit` > `Preferences` > `Add-Ons`.
3. Click on `Install` and select the downloaded zip file.
4. Enable the add-on by checking the checkbox next to its name.
5. Install dependencies by clicking the provided button.
   Optionally: If you are comfortable with blender's python environment and you'd prefer installing the dependencies yourself, you can use the requirements.txt file. 

## Usage

1. Open Blender and head to the Render Properties Tab.
2. Scroll down to find the SculptMate Panel.
3. If you have the dependencies installed, you can select an image of the character you'd like to generate a mesh of.
4. Click on Generate.

## Example Usage

![Samples](assets/usage.gif)

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests with your improvements or bug fixes.

## Third-Party Libraries

This project utilizes the following third-party libraries:

- **Library Name:** [PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization](https://shunsukesaito.github.io/PIFuHD/)
- **Author:** Saito, Shunsuke and Simon, Tomas and Saragih, Jason and Joo, Hanbyul
- **Year:** 2020
