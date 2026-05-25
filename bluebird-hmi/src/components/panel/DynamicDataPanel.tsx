import { CircularProgress } from "@mui/material";
import { useDynamicDataQuery } from "api/api";
import { useAppDispatch, useAppSelector } from "app/hooks";
import Dot from "components/panel/items/Dot";
import { useEffect } from "react";
import { update } from "slices/dynamicDataSlice";
import { selectIndividualSectorIds } from "slices/scenarioSlice";

export default function DynamicDataPanel(props: {
  pollingInterval: number;
  pauseApiCalls: boolean;
}) {
  const { pollingInterval, pauseApiCalls } = props;
  const dispatch = useAppDispatch();
  const sectorId = useAppSelector(selectIndividualSectorIds);
  const { data, isError, isLoading, isFetching } = useDynamicDataQuery(
    { sectorId: sectorId != null ? sectorId.join(";") : "" },
    {
      pollingInterval: pollingInterval,
      refetchOnReconnect: true,
      skip: pauseApiCalls || sectorId == null,
    },
  );
  useEffect(() => {
    if (data && data.exists) {
      dispatch(
        update({
          time: data.time,
          actions: data.actions,
          aircraft: data.aircraft,
          optimiserPlans: data.optimiser_plans,
          namedBays: data.named_bays,
          gameState: data.game_state,
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
