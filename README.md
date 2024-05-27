# Minimal Shazam Audio Identification on MongoDB

This is a minimal implementation of Shazam audio identification algorithm[^1] using MongoDB as the storage engine. The implementation is not fully optimised and is only for educational purposes.

## Requirements
- Python 3.10+
- MongoDB 7.0+

## Quick Start

1. Clone the repository.
2. Install the required packages.
```bash
pip install -r requirements.txt
```
3. Install [MongoDB](https://docs.mongodb.com/manual/installation/) and start the server at localhost.
4. Run the following command to benchmark the implementation. `--port` is optional and defaults to 28000. `--map` specifies the way to built peak pairs in the constellation maps. `wang` is the default value and is based on the paper. `delaunay` uses Delaunay triangulation to extract the pairs. 
```bash
python main.py /path/to/fingerprinted/audio/files /path/to/query/audio/files --port 28000 --map wang
```

[^1]: [An Industrial-Strength Audio Search Algorithm](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)