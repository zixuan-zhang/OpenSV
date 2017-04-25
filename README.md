## Introduction

## Folders

### android
Android screen unlock application based on signature verification

### cloud
A platform server which support signature authentication service using Thrift.
#### Usage
1. Generate Thrift Python depencencies
```
cd cloud/data
thrift --gen py opensv.thrift
cp -r gen-py ../src/
```
2. Start server
```
python main.py &
```

### cloud_c
Demo of opensv.thrift usage using C language

### cloud_java
Demo of opensv.thrift usage using Java language

### data
Some scripts.

### java
Signature verification system using Java

### python
Signature verification system using Python. This folder contains main code about research and experiments. `main.py` is the main entry of experiment.
