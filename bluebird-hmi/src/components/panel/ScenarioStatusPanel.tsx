import { useRunnerStatusQuery } from "api/api";
import { useAppDispatch, useAppSelector } from "app/hooks";
import Dot from "components/panel/items/Dot";
import { useEffect } from "react";
import { update } from "slices/scenarioSlice";

interface ScenarioStatusPanelProps {
  pollingInterval: number;
  pauseApiCalls: boolean;
  isScenarioModalOpen: boolean;
}

export function ScenarioStatusPanel(props: ScenarioStatusPanelProps) {
  const { pollingInterval, pauseApiCalls, isScenarioModalOpen } = props;

  const dispatch = useAppDispatch();

  const { data, isError, isLoading, isFetching } = useRunnerStatusQuery(
    undefined,
    {
      pollingInterval: pollingInterval,
      refetchOnReconnect: true,
      skip: pauseApiCalls,
    },
  );

  useEffect(() => {
    if (data && data.exists) {
      dispatch(
        update({
          exists: data.exists,
          category: data.category,
          scenario: data.scenario,
          iterations: data.iterations,
          running: data.running,
          evolve_period: data.evolve_period, // period evolved with each evolve call
          tick_frequency_period: data.tick_frequency_period, // frequency of evolve calls as a period in seconds
          kill: data.kill,
          reload: data.reload,
        }),
      );
    }
  }, [data, dispatch]);

  if (isLoading) {
    return <Dot backgroundColor="amber" />;
  }

  // reset store in case of error or missing data
  if (isError || !data || (data && !data.exists)) {
    if (!isScenarioModalOpen) {
      dispatch({ type: "resetStore/resetStore" });
    }
    return <Dot backgroundColor="red" />;
  }

  if (isFetching) {
    return <Dot backgroundColor="#0b0" />;
  }

  dispatch(
    update({
      exists: data.exists,
      category: data.category,
      scenario: data.scenario,
      iterations: data.iterations,
      running: data.running,
      evolve_period: data.evolve_period, // period evolved with each evolve call
      tick_frequency_period: data.tick_frequency_period, // frequency of evolve calls as a period in seconds
      kill: data.kill,
      reload: data.reload,
      is_validation_run: data.is_validation_run,
      validation_metadata: data.validation_metadata,
    }),
  );

  return <Dot backgroundColor={"#bbb"} />;
}
