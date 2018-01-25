## Generating Video

![](https://github.com/PositronicsLab/wild-robot/blob/master/simwise4d/img/sw4d-simulation-settings.png)
<i>Figure 1</i> - This dialog is found at World | Simulation Settings.  If Microsoft Video is selected for Compressor, refer to Figure 3, then the Animation Frame Rate | Rate field needs to be 15fps.  The Time field will auto-calculate based on the value in the Rate field.  In the Integration Step frame, if the Integration Step field is changed manually, it will recalculate the values in the Animation Frame Rate frame; however, the Steps per Frame field can be changed without recalculating the Animation Frame Rate fields and the Integration Step field will be automatically calculated.  Therefore, the Time and Integration values shown here were automatically calculated from the values in the Rate and Steps per Frame fields.

![](https://github.com/PositronicsLab/wild-robot/blob/master/simwise4d/img/sw4d-export-video.png)
<i>Figure 2</i> - For a comparable two minute simulation run, the total number of frames needs to be 1800 which is equal to 120s at 15fps.  For a 30s video, 15fps yields 450 frames.

![](https://github.com/PositronicsLab/wild-robot/blob/master/simwise4d/img/sw4d-video-compression.png)
<i>Figure 3</i> - Select Microsoft Video as the Compressor.  No other settings in this or in child dialogs need to be changed.  With these settings, the video will be synchronized to the simulation virtual time using this particular codec.
