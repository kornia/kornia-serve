# kornia-serve

This is a simple grpc set of microservices that can be used to serve kornia models.
It is based on the farm_ng grpc framework: https://github.com/farm-ng/farm-ng-core

## Installation

### Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the following commands in separate terminals.

### 1. Run the camera service

```bash
python -m camera
```

You should see the sent messages in the terminal and the connected clients.

### 2. Run the inference service

```bash
python -m inference
```

### 3. Run the printer client

```bash
python -m printer
```

You should see the received messages in the terminal with the image size.

## Optional: Re-generate the protobuf files

```bash
./generate_proto.sh
```
