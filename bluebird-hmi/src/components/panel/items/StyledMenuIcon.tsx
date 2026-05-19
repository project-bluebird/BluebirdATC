import { SvgIcon } from "@mui/material";
import { ReactNode } from "react";

interface StyledMenuIconProps {
  icon: ReactNode;
}

export const StyledMenuIcon = ({ icon }: StyledMenuIconProps) => {
  return <SvgIcon style={{ fontSize: "large", color: "grey" }}>{icon}</SvgIcon>;
};
