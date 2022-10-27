# Run main script on each *.param file in the param_files directory

python=python3
main=main.py

echo 
echo Attempting kinemetry with the following parameter files:
for file in ./param_files/*.param; do
	echo '\t'$file;
done

for file in ./param_files/*.param; do
	echo ;
	echo Running $main on $file;
	echo ;
	$python $main $file;
done
