

To run the encode the code Encodec Model, run the following code

```
python3 create_h5_dataset.py -audio-dir <audio directory path> --json-path <path of metadata.json> --target-dir <target directory path> --no-dac True --no-clap True
```


To run the encode the code Descript Model, run the following code

```
python3 create_h5_dataset.py -audio-dir <audio directory path> --json-path <path of metadata.json> --target-dir <target directory path> --no-encodec True --no-clap True
```
