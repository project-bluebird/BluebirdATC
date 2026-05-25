import { useAppSelector } from "app/hooks";
import DynamicDataPanel from "components/panel/DynamicDataPanel";
import SectorPanel from "components/panel/SectorPanel";
import StaticDataPanel from "components/panel/StaticDataPanel";
import { ScenarioStatusPanel } from "components/panel/ScenarioStatusPanel";
import { selectExists, selectIndividualSectorIds } from "slices/scenarioSlice";

interface MiniScenarioPanelPanelProps {
  isScenarioModalOpen: boolean;
}

export function MiniScenarioPanel(props: MiniScenarioPanelPanelProps) {
  const { isScenarioModalOpen } = props;
  const exists: boolean = useAppSelector(selectExists);
  const individualSectorIds = useAppSelector(selectIndividualSectorIds);

  return (
    <>
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: "0.5rem",
          padding: 2,
        }}
      >
        <ScenarioStatusPanel
          pollingInterval={250}
          pauseApiCalls={false}
          isScenarioModalOpen={isScenarioModalOpen}
        />
        {exists && individualSectorIds != null ? (
          <>
            <DynamicDataPanel pollingInterval={500} pauseApiCalls={false} />
            <StaticDataPanel />
          </>
        ) : (
          <></>
        )}
      </div>
      {exists && (
        <SectorPanel isRunning={false} shouldReloadEnvironment={false} />
      )}
    </>
  );
}
