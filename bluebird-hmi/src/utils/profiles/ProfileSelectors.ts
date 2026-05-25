import { ColourProfile, LineWidthProfile, OpacityProfile } from "utils/types";
import {
  presentationColourProfile,
  presentationOpacityProfile,
  radarColourProfile,
  radarOpacityProfile,
} from "utils/profiles/ColourProfiles";
import {
  presentationLineWidthProfile,
  radarLineWidthProfile,
} from "utils/profiles/LineWidthProfiles";

// Select a colourProfile based on a string
export function selectColourProfile(profileName: string): ColourProfile {
  switch (profileName) {
    case "presentation":
      return presentationColourProfile;
    default:
      return radarColourProfile;
  }
}

// Select a lineWidthProfile based on a string
export function selectLineWidthProfile(profileName: string): LineWidthProfile {
  switch (profileName) {
    case "presentation":
      return presentationLineWidthProfile;
    default:
      return radarLineWidthProfile;
  }
}

// Select an opacityProfile based on a string
export function selectOpacityProfile(profileName: string): OpacityProfile {
  switch (profileName) {
    case "presentation":
      return presentationOpacityProfile;
    default:
      return radarOpacityProfile;
  }
}
