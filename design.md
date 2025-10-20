Doodie Duty is an application intended to detect when a dog is in an area unsupervised and trigger actions based on that. Actions could be something like playing a sound, sending a notification, recording video, etc.

It will use a camera and image processing to detect the presence of the dog. It should also be able to detect humans so it can determine whether the dog is unsupervised or not. Python and opencv probably make the most sense to use, but open to suggestions for other technology.

The application should have a web interface that allows looking at the camera stream, reviewing previous events and configuring parameters. Might also be needed to train the vision model.

The application will be developed on a macbook, but will eventually be deployed to a separate device. Most likely a raspberry pi.

