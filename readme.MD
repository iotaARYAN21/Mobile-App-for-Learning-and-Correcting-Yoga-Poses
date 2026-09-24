# Steps to run the project

 cd AI-Pose-Estimation/
 ls
 /usr/bin/python3.12 -m venv venv
 source venv/bin/activate
 python -m pip install --upgrade pip
 python -m pip install "mediapipe==0.10.21"
 python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))"
 python main.py 
