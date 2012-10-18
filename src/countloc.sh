files=`find . -name \*.h -or -name \*.cpp -or -name \*.i -or -name \*.py -or -name \*.cu | grep -v "magneto_cuda.py" - | grep -v "magneto_cpu.py" - | grep -v "build/" -`
wcinfo=`cat $files | sed '/^\s*$/d' | wc`
loc=`echo $wcinfo | awk '{print $1}'`
bytes=`echo $wcinfo | awk '{print $3}'`
echo "$loc lines of code and $bytes bytes"
