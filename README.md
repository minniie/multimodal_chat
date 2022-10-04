# photochat
Princeton senior thesis


### Install requirements
```
pip install -r requirements.txt
```

### Run demo
```
CUDA_VISIBLE_DEVICES=${id} python -m demo.run_demo
```

### Train image retriever
```
sh sh/train_image_retriever.sh
```

### Train response generator
```
sh sh/train_response_generator.sh
```