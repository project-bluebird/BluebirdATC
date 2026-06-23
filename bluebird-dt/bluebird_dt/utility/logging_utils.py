import io
import json
import tarfile

import pandas as pd


def save_df_to_parquet_tar(df: pd.DataFrame, tar: tarfile.TarFile, file_name: str) -> None:
    """
    Save a dataframe to a parquet file into a tar archive.

    Parameters
    ----------
    df: DataFrame
        Panda dataframe to export from
    tar: TarFile
        The tar file of the log.
    file_name: str
        The name of the df
    """
    with io.BytesIO() as parquet_buffer:
        df.to_parquet(parquet_buffer, index=False, engine="pyarrow")
        parquet_info = tarfile.TarInfo(name=f"{file_name}.parquet")
        parquet_info.size = parquet_buffer.getbuffer().nbytes
        parquet_buffer.seek(0)
        tar.addfile(parquet_info, parquet_buffer)


def save_df_to_csv_tar(df: pd.DataFrame, tar: tarfile.TarFile, file_name: str) -> None:
    """
    Save a dataframe to a csv file into a tar archive.

    Parameters
    ----------
    df: DataFrame
        Panda dataframe to export from
    tar: TarFile
        The tar file of the log.
    file_name: str
        The file name to be saved
    """
    with io.BytesIO() as csv_buffer:
        df.to_csv(csv_buffer, index=False, float_format="%.9f", encoding="utf-8")
        csv_info = tarfile.TarInfo(name=f"{file_name}.csv")
        csv_info.size = csv_buffer.getbuffer().nbytes
        csv_buffer.seek(0)
        tar.addfile(csv_info, csv_buffer)


def save_json_to_tar(json_str: str, tar: tarfile.TarFile, file_name: str) -> None:
    """
    Save a json str to a json file into a tar archive.

    Parameters
    ----------
    json: str
        A json str
    tar: TarFile
        The tar file of the log.
    file_name: str
        The file name to be saved
    """
    json_bytes = json_str.encode("utf-8")
    json_info = tarfile.TarInfo(name=f"{file_name}.json")
    json_info.size = len(json_bytes)
    tar.addfile(json_info, io.BytesIO(json_bytes))


def read_tar_parquet_to_df(tar: tarfile.TarFile, file_name: str) -> pd.DataFrame:
    """
    Read a parquet file from a tar file and return as dataframe.

    Parameters
    ----------
    tar: TarFile, The tar file of the log.
    file_name: str, The specific parquet file to read

    Returns
    -------
    DataFrame, a panda dataframe.
    """
    try:
        file = tar.extractfile(f"{file_name}.parquet")
        if not file:
            raise FileNotFoundError(f"{file_name}.parquet not found in the log archive")
        return pd.read_parquet(file, engine="pyarrow")
    except KeyError:
        raise FileNotFoundError(f"{file_name}.parquet not found in the log archive") from None


def read_tar_csv_to_df(tar: tarfile.TarFile, file_name: str) -> pd.DataFrame:
    """
    Read a csv file from a tar file and return as dataframe.

    Parameters
    ----------
    tar: TarFile, The tar file of the log.
    file_name: str, The specific csv file to read

    Returns
    -------
    DataFrame, a panda dataframe.
    """
    try:
        file = tar.extractfile(f"{file_name}.csv")
        if not file:
            raise FileNotFoundError(f"{file_name}.csv not found in the log archive")
        return pd.read_csv(file)
    except KeyError:
        raise FileNotFoundError(f"{file_name}.csv not found in the log archive") from None


def read_tar_json_to_dict(tar: tarfile.TarFile, file_name: str) -> dict:
    """
    Read a json file from a tar file and return as dict.

    Parameters
    ----------
    tar: TarFile
        The tar file of the log.
    file_name: str
        The specific json file to read, minus the suffix

    Returns
    -------
    dict, A python dict.
    """
    try:
        file = tar.extractfile(f"{file_name}.json")
        if not file:
            raise FileNotFoundError(f"{file_name}.json not found in the log archive")
        return json.load(file)
    except KeyError:
        raise FileNotFoundError(f"{file_name}.json not found in the log archive") from None


def read_logs_from_tar(tar: tarfile.TarFile) -> dict[str, pd.DataFrame | dict]:
    """
    Read all the parquet or json files from a tarfile.

    Parameters
    ----------
    tar: tarfile.Tarfile
        path to tarfile containing logs

    Returns
    -------
    dict[str, pd.DataFrame]
        dictionary of resulting dataframes, keyed by the base name of
        the input file (e.g. "radar").
    """
    data = {}
    for m in tar.getmembers():
        if not m.isfile():
            continue
        if m.name.endswith(".parquet"):
            key = m.name.removesuffix(".parquet")
            data[key] = read_tar_parquet_to_df(tar, key)
        elif m.name.endswith(".json"):
            key = m.name.removesuffix(".json")
            data[key] = read_tar_json_to_dict(tar, key)
    return data
