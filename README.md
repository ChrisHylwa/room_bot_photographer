Remember to run `catkin_make` and `source devel/setup.bash` after getting this onto the Turtlebot terminals.

Should be able to run this using `roslaunch photographer photographer.launch`.

If not, run the python code in the scripts folder using `rosrun`.

Note: The package code is a bit broken at the moment, so in order to run the script, you'll have to navigate to the scripts folder, and then run `python photographer.py`. The reasons for this are unclear, as the exact bug preventing rosrun from working is pretty inscrutable at the moment.

Note: USB Cam feature is still not implemented in this current build, so make sure to use the digital camera and save those images seperately
