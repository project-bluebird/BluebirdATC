import { CircularProgress } from "@mui/material";
import { useStaticDataQuery } from "api/api";
import { useAppDispatch, useAppSelector } from "app/hooks";
import Dot from "components/panel/items/Dot";
import { useEffect } from "react";
import { update } from "slices/staticDataSlice";

export default function StaticDataPanel() {
  const dispatch = useAppDispatch();

  const { data, isError, isLoading, isFetching } = useStaticDataQuery(
    undefined,
    {
      refetchOnMountOrArgChange: true,
      refetchOnReconnect: true,
    },
  );

  useEffect(() => {
    if (data && data.exists) {
      dispatch(
        update({
          exists: data.exists,
          scenarioName: data.scenario_name,
          bayNames: data.bay_names,
          fixes: data.fixes,
          sectors: data.sectors,
          projection_centre: data.projection_centre,
        }),
      );
    }
  }, [data, dispatch]);

  if (isError) {
    return <Dot backgroundColor="red" />;
  }
  if (isLoading) {
    return <Dot backgroundColor="amber" />;
  }
  if (!data) {
    return <Dot backgroundColor="red" />;
  }
  if (!data.exists) {
    return <CircularProgress size={10} />;
  }

  if (isFetching) {
    return <Dot backgroundColor="#0b0" />;
  }

  return <Dot backgroundColor={"#bbb"} />;
}
