echo "Launching test_classification.py ..."
dd=$(date +"%Y%m%d_%H%M%S")
name=$(basename $1)
if [ -z "$2" ]
then
	video_name="output_${dd}"
else
	video_name=$2
	if [ -f "./output/$name/$video_name.mp4" ]
	then
		video_name="$2_${dd}"
	fi
fi
echo $video_name

nohup python test_classification.py $1 $video_name > Log_sub_${dd}.txt 2>&1 &

#check if the directory creates
while [ ! -d "./output/$name" ];do
	sleep 60
done
#check the video and csv file are created
#then delete Image folder and interval csv file
cd ./output/$name
FILE=./$video_name.mp4
while [ ! -f "$FILE" ];do
	sleep 120
	echo "The program is working..."
done
rm prediction_subject.csv
rm -r Image/
mv ../../Log_sub_${dd}.txt ./Log_sub_${dd}.txt
grep "number\|overall" ./Log_sub_${dd}.txt
cd ../../
echo "This subject is done"
