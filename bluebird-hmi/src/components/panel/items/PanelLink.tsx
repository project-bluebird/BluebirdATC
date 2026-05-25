import PanelButton from "components/panel/items/PanelButton";
import PanelProgress from "components/panel/items/PanelProgress";
import { ReactElement, useCallback, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

interface PanelLinkProps {
  to: string;
  text: string;
  waitingText?: string;
  icon: ReactElement;
  disabled?: boolean;
  target?: string; // e.g., "_blank" for opening in a new tab
  onClick?: () => void;
}

// Panel links are PanelButtons used for routing
export default function PanelLink(props: PanelLinkProps) {
  const [isWaiting, setWaiting] = useState(false);
  const navigate = useNavigate();

  const handleWaitAndNavigate = useCallback(async () => {
    if (props.onClick && !isWaiting) {
      setWaiting(true);
      await props.onClick();
      navigate(props.to);
      setWaiting(false);
    }
  }, [props, navigate, isWaiting]);

  // only call onClick if it exists
  const handleClick = (event) => {
    if (props.onClick) {
      // prevent default link behaviour until we have response from API
      event.preventDefault();
      handleWaitAndNavigate();
    }
  };

  const waitingText = props.waitingText ? props.waitingText : "Waiting";
  if (isWaiting) {
    return <PanelProgress text={waitingText} />;
  }
  return (
    <Link to={props.to} target={props.target} onClick={handleClick}>
      <PanelButton
        text={props.text}
        icon={props.icon}
        disabled={props.disabled}
      />
    </Link>
  );
}
