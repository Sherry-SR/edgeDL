#!/usr/bin/env bash
DATADIR="/home/SENSETIME/shenrui/data/pelvis"
SAVEDIR="/home/SENSETIME/shenrui/data/pelvis_resampled(0.8,0.8,0.8)"
FILES=$DATADIR/*
for file in $FILES
do
	if [[ ($file == *.nii.gz) || ($file == *.nii)]]
	then
		echo Current image path is: $file
		filename=$(basename ${file%.nii*})
		echo Filename is $filename
		if [[ $filename == *_label ]]
		then
			mri_convert -vs 0.8 0.8 0.8 -rt nearest -odt short -ns 1 $file $SAVEDIR/$filename.nii.gz
		else
			mri_convert -vs 0.8 0.8 0.8 -rt interpolate -odt short -ns 1 $file $SAVEDIR/$filename.nii.gz
		fi
	fi
done
