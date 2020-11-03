#!/bin/zsh
# This script should be run from the EEG_CNNs directory.

if [ -d "./experiments" ]
then
    if [ -d "./experiments/CNN_saves" ]
    then
        SAVEDIR="./experiments/CNN_saves"
    fi
else
    SAVEDIR="./TEMP_save"
fi

if [ ! -d "./outputs" ]
then
    mkdir ./outputs
fi

for DIM in 1 2 
do
    for PARTICIPANT in 19
    do
        for MAX_EPOCH in 30
        do
            for BATCHSIZE in 8
            do
                for LEARNING_RATE in 001
                do
                    for MODE in bc
                    do
                        for FILE in single resample all_data
                        do
                            for CATEGORY in "speaking" "stimuli" "thinking"
                            do
                                OUTFILENAME="${DIM}D_${FILE}_P$PARTICIPANT-$MODE-LR$LEARNING_RATE-ME$MAX_EPOCH-B$BATCHSIZE-$CATEGORY.txt"
                                FINDIRNAME="$SAVEDIR/${DIM}D_${FILE}_P$PARTICIPANT-$MODE-LR$LEARNING_RATE-ME$MAX_EPOCH-B$BATCHSIZE-$CATEGORY"
                                echo "TRAINING:" > ./outputs/$OUTFILENAME
                                python ./${DIM}D_$FILE.py -lr 0.$LEARNING_RATE -me $MAX_EPOCH -bs $BATCHSIZE -$MODE -ts -c $CATEGORY>>./outputs/$OUTFILENAME
                                OUTDIR1=$(less ./outputs/$OUTFILENAME|grep "used_settings.txt" | cut -d'/' -f 3)
                                echo "\n\n\n\nTESTING:" >> ./outputs/$OUTFILENAME
                                python ./${DIM}D_$FILE.py -lr 0.$LEARNING_RATE -me $MAX_EPOCH -bs $BATCHSIZE -$MODE -ds -c $CATEGORY -l "${OUTDIR1}">>./outputs/$OUTFILENAME
                                echo "$OUTDIR1 --> $OUTFILENAME"
                                mv $SAVEDIR/$OUTDIR1 $FINDIRNAME
                                cp ./outputs/$OUTFILENAME $FINDIRNAME/TOTAL_RUN_OUTPUT.txt
                            done
                        done
                    done
                done
            done
        done
    done
done