# animeface
* organizes the data from the anime face dataset into one large, flat folder
* data from http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/
* which i found here: https://www.kaggle.com/mylesoneill/tagged-anime-illustrations/home
* the data is ~160x160 images of anime faces, containing 14490 images
* original structure was a directory of images for each character. Each character directory contained some csv file, but I ignored that
* these scripts flatten the character directories into one large image directory
* find_size_bounds determines the maximum and minimum width and height across all images