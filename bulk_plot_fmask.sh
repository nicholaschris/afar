for line in $(cat $1)
do 
echo "$line"
python plot_fmask.py /storage/Nicholas/Data/ $line /storage/Nicholas/Data/
done 
