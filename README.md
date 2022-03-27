# Torch object detector re-trainer
Using PyTorch's Object Detection API to re-train and adapt a model to our purposes from a COCO dataset.
It also exploits FiftyOne capabilities to improve dataset inspection and analysis.
## Usage:

Build the docker image:

```
$ docker build -t object-detector . 
```


Navigate to the folder where your dataset resides:

```
 cd <your dataset path> 
```

Run the docker image and use the provided link to localhost

```
docker run -it --network host -v $(pwd):/dataset --gpus all --shm-size 8G -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:24 object-detector jupyter notebook --allow-root
```

Once Jupyter Notebook is running, use the browser to access the object-detector.ipynb notebook under the /scripts folder
