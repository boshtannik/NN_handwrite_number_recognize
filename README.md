# NN_handwrite_number_recognize
Fork of simple neural network to recognize handwrite digits, took from MNIST dataset

(All consider using `python3` - when `python` is mentioned)

To work with neural network - it must have mnist dataset installed.
Two options of dataset available. So you can install it in few ways.

1 - To install large dataset for training (60 000 examples) - run ```bash ./install_data.sh``` - Will download datasets
or
2 - To install smaller dataset for training (100 examples) - run ```bash ./install_data_small.sh```	- Will download datasets

Then you can run ```python neural_network.py``` - It will open files to train NN, and will train it.
I would reccoment to use ```pypy``` instead of ```python``` - Special python interpreter with JIT compiler - which will make train a LOT faster

Also you can visually check - if the neural network did good train. For this purpose `tester.py` - is provided.

To run that, you need to have pygame library installed.

Here is the instruction i offer to follow to get it installed:
1 - I recommend you to use virtual environment package. To install it - use `sudo apt install python-venv`
2 - then create your local isolated python environment. To create it - use `python -m venv venv` - It will create directory venv in your current directory
3 - Use just installed local python environment. To use it - run `source ./venv/bin/activate` - You will see (venv) prefix in your prompt
4 - Install all required dependencies of tester.py into local environment. To install them - run `pip install -r requirements.txt` - It will read requirements file end then install dependencies.
5 - Run it ```python tester.py``` - It will take some time to load your trained neural network data, and then immediately will show you the GUI

GUI:
	Button `Next` - Will read next line from test dataset, load it onto canvas, and then will make the classification
	Button `Recognize` - Will send data from canvas to neural network - to make classification
	Button `Clear`	- Will make canvas clean
	Also you can draw onto canvas - and then immediately send drawn data into NN, and make the classification of drawn symbol.
