for file in "$1"/*; do
	echo ' '
	echo 'Running kinemetry on : '
	echo '\t'$file
	echo ' '
	python3 main.py $file
done
