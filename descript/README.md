

To run the encode the code Encodec Model, run the following code

```
python3 create_h5_dataset.py -audio-dir <audio directory path> --json-path <path of metadata.json> --target-dir <target directory path> --no-dac True --no-clap True
```


To run the encode the code Descript Model, run the following code

```
python3 create_h5_dataset.py -audio-dir <audio directory path> --json-path <path of metadata.json> --target-dir <target directory path> --no-encodec True --no-clap True
```


If you are running on AI mega center cluster, please install the following dependencies

```
pip install librosa encodec h5py audio_metadata tinytag audiocraft

pip install torch==2.1.0

pip install descript-audio-codec

pip install torchvision==0.16.0
```