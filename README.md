# multimodal_chat
Princeton senior thesis by Min Young Lee

### Install requirements
```
sh setup.sh
```

### Run demo
```
CUDA_VISIBLE_DEVICES=${id} python -m demo.run_demo
```

### Train and evaluate image retriever
```
sh sh/train_image_retriever.sh
sh sh/eval_image_retriever.sh
```

### Train and evaluate response generator
```
sh sh/train_response_generator.sh
sh sh/eval_response_generator.sh
```
