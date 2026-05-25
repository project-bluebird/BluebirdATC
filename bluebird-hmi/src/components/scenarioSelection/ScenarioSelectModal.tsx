import { useState } from "react";
import { CategorySelect } from "components/scenarioSelection/CategorySelect";
import { ScenarioSelect } from "components/scenarioSelection/ScenarioSelect";
import { ScenarioLoad } from "components/scenarioSelection/ScenarioLoad";
import { useLoadMutation } from "api/api";
import { useAppDispatch } from "app/hooks";

type ScenarioSelectModalProps = {
  isOpen: boolean;
  onClose: () => void;
};

export function ScenarioSelectModal({
  isOpen,
  onClose,
}: ScenarioSelectModalProps) {
  const dispatch = useAppDispatch();
  const [load] = useLoadMutation();
  const [step, setStep] = useState(0);

  const [scenarioCategory, setScenarioCategory] = useState(null);
  const [scenarioName, setScenarioName] = useState(null);

  if (!isOpen) return null;

  const resetStore = () => {
    dispatch({ type: "resetStore/resetStore" });
  };

  const loadScenario = () => {
    console.log("in loadScenario with ", scenarioCategory, scenarioName);
    //   resetStore();
    if (scenarioCategory === null || scenarioName === null) {
      return;
    }
    load({
      category: scenarioCategory,
      scenarioName: scenarioName,
    });
    onClose();
  };

  const goBack = () => setStep((s) => Math.max(0, s - 1));
  const goNext = () => setStep((s) => s + 1);

  const handleCancel = () => {
    setStep(0);
    onClose();
  };

  const handleBack = () => {
    goBack();
  };

  const handleSelectCategory = (category: string) => {
    setScenarioCategory(category);
    goNext();
  };

  const handleSelectScenario = (scenario: string) => {
    setScenarioName(scenario);
    goNext();
  };

  const handleLoadScenario = () => {
    console.log("LOADING SCENARIO");
    resetStore();
    loadScenario();
    setStep(0);
  };

  return (
    <div className="modal-backdrop">
      <div className="modal">
        {step === 0 && (
          <CategorySelect
            handleSelectCategory={handleSelectCategory}
            handleCancel={handleCancel}
          />
        )}

        {step === 1 && (
          <ScenarioSelect
            category={scenarioCategory}
            handleSelectScenario={handleSelectScenario}
            handleBack={handleBack}
            handleCancel={handleCancel}
          />
        )}

        {step === 2 && (
          <ScenarioLoad
            category={scenarioCategory}
            scenarioName={scenarioName}
            handleLoad={handleLoadScenario}
            handleBack={handleBack}
            handleCancel={handleCancel}
          />
        )}
      </div>
    </div>
  );
}
