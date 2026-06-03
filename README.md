# Dependecies
Actually running the script requires you to get the OpenVDB library for C++. There are 2 proper ways to put the project together:

## Manual installation
You can get the library manually from the official repo at https://github.com/AcademySoftwareFoundation/openvdb.
After that chuck it right inside this project's directory so it looks something like this:

```
PlanetaryLandscapeGenerator/
├── Dependencies/
│   └── ...
├── DiplomaWork/
│   └── ...
├── openvdb/
│   └── ...
└── CMakeLists.txt
```
After that the CMake compiler should do all the rest for you, possibly.

## Git Submodules
For a more comprehensive guide, I suggest checking out https://git-scm.com/book/en/v2/Git-Tools-Submodules

What should probably be enough to start the project though is the following:
First, clone the main project. After that you will need to install the openvdb library represented as a git submodule:
```bash
$ cd PlanetaryLandscapeGenerator
$ git submodule init
$ git submodule update
```

With any luck this should result in a working project.

Or you could just do this:
```bash
$ git clone --recurse-submodules https://github.com/shleble/PlanetaryLandscapeGenerator.git
```
Fancy, huh?