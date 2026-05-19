import ErrorIcon from "@mui/icons-material/Error";
import MapIcon from "@mui/icons-material/Map";
import MenuList from "@mui/material/MenuList";
import { AirspaceRead, useCompleteEnvironmentQuery } from "api/api";
import { useAppDispatch, useAppSelector } from "app/hooks";
import PanelData from "components/panel/items/PanelData";
import PanelProgress from "components/panel/items/PanelProgress";
import PanelSelect from "components/panel/items/PanelSelect";
import { useCallback, useEffect, useState } from "react";
import { changeSectorId } from "slices/scenarioSlice";

interface SectorPanelProps {
  isRunning: boolean;
  shouldReloadEnvironment: boolean;
}

/**
 * Gives a list of all the keys of the individual sectors in the airspace
 * @param { ApiAirspace | undefined } airspace - The airspace representation to be displayed
 * @returns { string[] | undefined } - List of the keys of the individual sectors, which can be used to query airspace.sector, or undefined if airspace is undefined
 */
function individual_sectors(
  airspace: AirspaceRead | undefined,
): string[] | undefined {
  if (!airspace) {
    return undefined;
  }

  const airspace_configs = airspace.airspace_configuration;

  return Object.keys(airspace_configs)
    .map((key) => airspace_configs[key])
    .flat()
    .sort();
}

/**
 *
 * @param { ApiAirspace | undefined } airspace - The airspace representation to be displayed
 * @returns { string | undefined } - List of currently available list of bandbox configurations, or undefifned if ApiAirspace is undefined.
 */
function current_bandboxed_sectors(
  airspace: AirspaceRead | undefined,
): string[] | undefined {
  if (!airspace) {
    return undefined;
  }
  return Object.keys(airspace.airspace_configuration).sort();
}

/**
 * This function returns the string key of the bandboxed configuration currently used in the airspace given the list of sectors selected, or "Custom" if it is a custom, non active, configuration.
 *
 * @param { ApiAirspace | undefined } airspace - The airspace representation to be displayed
 * @param { string[] } selected_sectors - The list of sector keys that need to be matched to a bandbox configuration
 * @returns { string } - The key for the bandbox configuration or "Custom" if none is found, therefore the provided list is custom.
 */
function current_bandbox_config(
  airspace: AirspaceRead | undefined,
  selected_sectors: string[],
): string {
  if (!airspace) {
    return "";
  }

  const airspace_configs = JSON.parse(
    JSON.stringify(airspace.airspace_configuration),
  );
  const selected_sectors_copy = JSON.parse(
    JSON.stringify(selected_sectors),
  ).sort();
  const result = Object.keys(airspace_configs).find(
    (key) =>
      airspace_configs[key].sort().join(",") ===
      selected_sectors_copy.join(","),
  );
  return result ? result : "Custom";
}

/**
 * This function returns the string key of the bandboxed sector containing the most individual sectors.
 *
 * @param { ApiAirspace | undefined } airspace - The airspace representation to be displayed
 * @returns { string } - The key for the bandbox configuration that contains the largest number of sectors.
 */
function largest_bandbox_config(
  airspace: AirspaceRead | undefined,
): string | undefined {
  if (!airspace) return undefined;

  return Object.entries(airspace.airspace_configuration).reduce(
    (best, [key, sectors]) =>
      sectors.length > (best[1] as string[]).length ? [key, sectors] : best,
  )[0] as string;
}

export default function SectorPanel(props: SectorPanelProps) {
  const { isRunning, shouldReloadEnvironment } = props;

  const dispatch = useAppDispatch();
  const [sector, setSector] = useState<string[]>([]);
  const { data, isError, isLoading, refetch } = useCompleteEnvironmentQuery(
    undefined,
    {
      refetchOnMountOrArgChange: true,
      refetchOnReconnect: true,
      pollingInterval: 12000,
    },
  );

  const toggleSectors = useCallback(
    (individualSectors: string[]) => {
      setSector(individualSectors);

      if (individualSectors.length > 0) {
        const new_bandbox_config = current_bandbox_config(
          data.airspace,
          individualSectors,
        );
        dispatch(
          changeSectorId({
            individualSectorIds: individualSectors,
            bandboxSectorId:
              new_bandbox_config !== "Custom" ? new_bandbox_config : null,
          }),
        );
      } else {
        dispatch(
          changeSectorId({ individualSectorIds: null, bandboxSectorId: null }),
        );
      }
    },
    [data, dispatch],
  );

  useEffect(() => {
    if (shouldReloadEnvironment) {
      refetch();
    }
  }, [shouldReloadEnvironment, refetch]);

  useEffect(() => {
    if (
      data &&
      sector.find((id) =>
        Object.values(data.airspace.airspace_configuration).flat().includes(id),
      ) === undefined
    ) {
      const bandbox_config =
        largest_bandbox_config(data.airspace) ??
        Object.keys(data.airspace.airspace_configuration)[0];
      const sectors = data.airspace.airspace_configuration[bandbox_config];

      if (Object.keys(data.airspace.sectors).length > 0) {
        setSector(sectors);
        toggleSectors(sectors);
      }
    }
  }, [data, dispatch, sector, toggleSectors]);

  const handleSectorSelection = (event) => {
    toggleSectors(event.target.value);
  };

  const handleProfileSelection = (event) => {
    const sectors = data.airspace.airspace_configuration[event.target.value];
    toggleSectors(sectors);
  };

  if (isError) {
    return (
      <MenuList>
        <PanelData
          icon={<ErrorIcon />}
          text={"An error occurred fetching environment data."}
        />
      </MenuList>
    );
  }
  if (isLoading) {
    return <PanelProgress text={"Loading environment data"} />;
  }
  if (!data) {
    return (
      <MenuList>
        <PanelData icon={<ErrorIcon />} text={"Environment data is missing."} />
      </MenuList>
    );
  }

  const ind_sectors = individual_sectors(data.airspace);
  const bandboxed_sectors = current_bandboxed_sectors(data.airspace);
  const current_bb_config = current_bandbox_config(data.airspace, sector);

  if (data && Object.keys(data.airspace.sectors).length === 0) {
    return (
      <MenuList>
        <PanelData icon={<MapIcon />} text={`No sectors in Environment.`} />
      </MenuList>
    );
  } else if (data && Object.keys(data.airspace.sectors).length === 1) {
    return (
      <MenuList>
        <PanelData
          icon={<MapIcon />}
          text={`Sector: ${Object.keys(data.airspace.sectors)[0]}`}
        />
      </MenuList>
    );
    // not sure how it's possible that data is defined and sector is undefined
    // but it does seem to happen for the odd split second (maybe setSector not called quickly enough?)
  } else if (data && sector !== undefined) {
    return (
      <MenuList>
        <PanelSelect
          icon={<MapIcon />}
          id="sector-select"
          label="All individual sectors"
          labelId="sector-select-label"
          value={sector}
          onChange={handleSectorSelection}
          choices={ind_sectors}
          isRunning={isRunning}
        />
        <PanelSelect
          icon={<MapIcon />}
          id="sector-select"
          label="Currently active bandboxed sectors"
          labelId="sector-select-label"
          value={current_bb_config}
          onChange={handleProfileSelection}
          choices={bandboxed_sectors}
          isRunning={isRunning}
        />
      </MenuList>
    );
  } else {
    return <>Sector data is missing.</>;
  }
}
