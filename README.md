## Installation
To begin, create a virtual environment using Python 3.8+ with PyTorch 1.9.1 and CUDA toolkit 11.3.

```bash
conda env create -f environment.yaml
```

Download the dataset from this [link](https://pan.baidu.com/s/1byeUjoqzJCfHEmUZMeNRoQ?pwd=bsjp) and unzip it into the "datasets" folder.

Compile third-party libraries in the "third_party" folder:

```bash
python setup.py install
```

Copy the path of the compiled `svo.cpython-xx-xxx.so` file into the torch.classes.load_library function in `main.py`:

```python
torch.classes.load_library("third_party/build/lib.linux-x86_64-3.8/svo.cpython-38-x86_64-linux-gnu.so")
```

## Run
Execute the following commands to run the code:

```bash
python main.py config/replica/office0.yaml
python main.py config/replica/office1.yaml
# ...
```

## Result
The results will be saved in the "out" folder.