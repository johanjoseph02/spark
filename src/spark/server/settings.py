from __future__ import annotations

import os
from dataclasses import dataclass, field

from spark.server.pricing import PricingConfig


@dataclass(frozen=True)
class Settings:
    db_path: str = "spark.db"
    lots_dir: str = "configs/lots"
    device_token: str = ""  # bearer token edge agents send with reports; empty = open
    admin_token: str = ""  # bearer token for operator actions; empty = open
    snapshot_every_s: float = 60.0  # at most one occupancy snapshot per lot per interval
    pricing: PricingConfig = field(default_factory=PricingConfig)
    gtfs_dir: str = ""

    @classmethod
    def from_env(cls, **overrides: str) -> Settings:
        env = os.environ
        return cls(
            db_path=overrides.get("db_path") or env.get("SPARK_DB", "spark.db"),
            lots_dir=overrides.get("lots_dir") or env.get("SPARK_LOTS", "configs/lots"),
            device_token=env.get("SPARK_DEVICE_TOKEN", ""),
            admin_token=env.get("SPARK_ADMIN_TOKEN", ""),
            gtfs_dir=overrides.get("gtfs_dir") or env.get("SPARK_GTFS", ""),
        )
