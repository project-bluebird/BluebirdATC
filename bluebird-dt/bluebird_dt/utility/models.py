from typing import Annotated

from pydantic import Field

BandboxSectorName = Annotated[
    str,
    Field(
        title="Bandbox sector name",
        description="Name of a bandbox sector.",
        examples=["springfield"],
    ),
]

IndividualSectorName = Annotated[
    str,
    Field(
        title="Individual sector name",
        description="Name of an individual sector.",
        examples=["sector_1", "sector_2"],
    ),
]
