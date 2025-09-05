# src/utils/models.py

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

@dataclass(frozen=True)
class DatasetMetadata:
    """
    A standardized, immutable data structure for holding the results of a crawl.

    'frozen=True' means instances of this class cannot be changed after creation.
    This is a best practice for data objects that are passed through a pipeline,
    as it prevents accidental modification.
    """
    subject: str
    modality: str
    source: str
    attributes: list[str]
    files: list[Path]

    # The timestamp is automatically generated with timezone info when the object is created.
    # 'init=False' means we don't have to provide it when we create an instance.
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        init=False
    )

    @property
    def count(self) -> int:
        """
        The actual number of files successfully downloaded. This is a read-only
        property derived from the length of the files list, so it can't be wrong.
        """
        return len(self.files)
    
class Crawler(Protocol):
    """
    Defines the standard interface (the "contract") that all crawlers must follow.

    Our pipeline manager will use this protocol to interact with any crawler
    without needing to know its specific implementation details. It just needs to
    adhere to this "shape".
    """

    def scrape(
        self,
        subject: str,
        limit: int,
        attributes: list[str],
        output_dir: Path
    ) -> DatasetMetadata | None:
        """
        The main method for any crawler. It scrapes data based on the given
        parameters and returns a DatasetMetadata object on success.

        Args:
            subject: The main topic to search for (e.g., 'Tzuyu', 'dog').
            limit: The desired number of items to download.
            attributes: A list of extra qualifiers (e.g., ['face', 'bark']).
            output_dir: The directory where downloaded files should be saved.

        Returns:
            A DatasetMetadata object if the crawl was successful.
            None if the crawl failed.
        """
        ... # The "..." here is the literal syntax for defining a protocol method.    