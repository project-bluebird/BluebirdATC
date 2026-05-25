import { Box, MenuList, Slider, Typography } from "@mui/material";
import { usePauseMutation, useRewindMutation } from "api/api";
import { useAppSelector } from "app/hooks";
import DrawerButton from "components/radarDrawer/DrawerButton";
import { useEffect, useState } from "react";
import { selectTime } from "slices/dynamicDataSlice";

const TimeSlider = () => {
  const endTime = useAppSelector(selectTime);
  const [rewind] = useRewindMutation();
  const [pause] = usePauseMutation();

  const timeStringToSeconds = (timeString) => {
    return new Date(timeString).getTime() / 1000;
  };

  const secondsToTimeString = (seconds) => {
    const date = new Date(seconds * 1000);
    return date.toISOString();
  };

  const [displayValue, setDisplayValue] = useState(endTime);
  const [sliderValue, setSliderValue] = useState(600);

  useEffect(() => {
    setDisplayValue(endTime);
  }, [endTime]);

  const endDateTimeInSeconds = timeStringToSeconds(endTime + "Z");
  const totalDurationInSeconds = 3600;

  const handleChange = (event) => {
    const value = event.target.value;
    const interpolatedSeconds =
      endDateTimeInSeconds + (value / 600 - 1) * totalDurationInSeconds;
    const interpolatedTime = secondsToTimeString(interpolatedSeconds);
    setDisplayValue(interpolatedTime.replace("T", " ").substring(0, 19));
    setSliderValue(value);
  };

  const handleMouseDown = () => {
    pause();
  };

  const submitRewind = () => {
    setSliderValue(600);
    rewind({
      newTime: displayValue,
    });
  };

  return (
    <MenuList
      disablePadding
      sx={{
        mt: 0.75,
        alignItems: "center",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <Typography>{"Rewind to"}</Typography>
      <Typography>{displayValue}</Typography>
      <Slider
        value={sliderValue}
        min={0}
        max={600}
        step={1}
        onChange={handleChange}
        onMouseDown={handleMouseDown}
        sx={{ width: "80%" }}
      />
      <DrawerButton text={"Rewind"} onClick={() => submitRewind()} />
    </MenuList>
  );
};

export default TimeSlider;
