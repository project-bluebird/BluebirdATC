import HistoryToggleOffIcon from "@mui/icons-material/HistoryToggleOff";
import { useEvolveMutation } from "api/api";
import { useAppSelector } from "app/hooks";
import PanelButton from "components/panel/items/PanelButton";

interface PanelTimeJumpProps {
  text: string;
  timeDelta: number;
}

export default function PanelTimeJump({ text, timeDelta }: PanelTimeJumpProps) {
  const [evolve] = useEvolveMutation();

  return (
    <PanelButton
      text={text}
      icon={<HistoryToggleOffIcon />}
      onClick={() => evolve({ timeDelta })}
    />
  );
}
