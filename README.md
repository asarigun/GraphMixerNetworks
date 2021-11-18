# Graph Nasreddin Networks

> *In this model, we try to use MLP-Mixer in Graph Neural Networks. The motivation for this project is to replace Transformers and Message Passing with MLP-Mixers in Graph Neural Nets*

# Preprint

**Preprint will be published soon!**
## Overview

* ```geometric_linear.py```: **Linear Layer from PyG Source Code** for Graph Nasreddin Networks
* ```gmn_layer.py```: **Graph Nasreddin Layer** 
* ```gmn_train_zinc.py```: **Graph Nasreddin Network** Training on ZINC Dataset

## Usage
```bash
python gmn_train_zinc.py
```

## License

[MIT](LICENSE)

## Acknowledgement

The name of the **Nasreddin** coming from Anatolian figure Nasreddin Hodja's story called **'What if it happens?'**. Also, while doing benchmarking, we use the **[PNA](https://arxiv.org/pdf/2004.05718.pdf)** paper implementation in **[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pna.py)**. Special thanks to authors for sharing code!