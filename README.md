wget http://pjreddie.com/media/files/yolo.weights
rosrun yolo_tf convert_weights yolo.weights /path/to/yolo_tf/data/yolo.ckpt

wget http://pjreddie.com/media/files/tiny-yolo-voc.weights
rosrun yolo_tf convert_weights tiny-yolo-voc.weights /path/to/yolo_tf/data/tiny-yolo-voc.ckpt

If the script crashes due to out of memory, try the following.
echo 1 > /proc/sys/vm/drop_caches
and/or
echo 2 > /proc/sys/vm/drop_caches  

# yolo_tf

## How to use Recognition(書き途中)
### yolo.py
1. yolo9000のネットワークを使用する
   1. [ココ](https://drive.google.com/open?id=1CHHccYks0Mgf2NGUDDFKIG_g6V8Il_QN) から 9k.names, 9k.tree, yolo9000.ckpt.data, yolo9000.ckpt.mate, yolo9000.ckpt.index の計5つのデータをダウンロードしてくる。yolo_tf/data/の中にでも入れておく  
   
   2. 自前で学習させた.names, .ckpt-* の４つを用意する  
   
   3. 以下のコマンドを実行してノードを起動  
      ```bash
      $ rosrun yolo_tf yolo.py --ckpt ~/..適当なパス../yolo9000.ckpt --names ~/..適当なパス../9k.names --tree ~/..適当なパス../9k.tree  --ckpt1 ~/..適当なパス../test.ckpt-12345 --names1 ~/..適当なパス../test.names --type1 classifier
      ```
      imageのりマッピングを忘れずに(xtion:「image:=/camera/rgb/image_rect_color」, HSR:「image:=/hsrb/head_rgbd_sensor/rgb/image_rect_color」)
2. yolov3のネットワークを使用する
   1. [ココ](https://drive.google.com/open?id=1CHHccYks0Mgf2NGUDDFKIG_g6V8Il_QN) から yolov3.ckpt.* の計3つのデータをダウンロードしてくる。yolo_tf/data/の中にでも入れておく  
   
   2. 自前で学習させた.names, .ckpt-* の４つを用意する  
   
   3. 以下のコマンドを実行してノードを起動  
      ```bash
      $ rosrun yolo_tf yolo.py --ckpt /..適当なパス../yolov3.ckpt --names /..適当なパス../coco.names --ckpt1 /..適当なパス../test.ckpt-12345 --names1 /..適当なパス../test.names --type1 classifier
      ```
      imageのりマッピングを忘れずに(xtion:「image:=/camera/rgb/image_rect_color」, HSR:「image:=/hsrb/head_rgbd_sensor/rgb/image_rect_color」)
   
