
ps -elf | grep " python "|grep -v "grep"|while read l
do
    pid=`echo $l | cut -f 4 -d " "`
    kill -9  $pid
    echo $l
done
