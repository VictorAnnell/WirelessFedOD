module purge && \
module load Python/3.11.5-GCCcore-13.2.0 && \
if [[ -d venv ]]
then
	source venv/bin/activate
else
	#virtualenv --system-site-packages venv && \
	python -m venv venv && \
	source venv/bin/activate && \
	pip install -r alvis-requirements.txt
fi
