# yolo_ros
ROS package implementing [YOLO](https://pjreddie.com/yolo/) with TensorFlow.

## How to use
* Download 9k.names, 9k.tree, yolo9000.ckpt.data, yolo9000.ckpt.meta, and yolo9000.ckpt.index from [here](https://drive.google.com/open?id=1CHHccYks0Mgf2NGUDDFKIG_g6V8Il_QN).

* Run the node as follows:
```bash
      $ rosrun yolo_ros yolo.py --ckpt /path/to/file/yolo9000.ckpt --names /path/to/file/9k.names --tree /path/to/file/9k.tree
```

* It subscribes 'image' and publishes the result to 'objects'. It also provides a ROS service 'detect_objects' that processes images on demand.
