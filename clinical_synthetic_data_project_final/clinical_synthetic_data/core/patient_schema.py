from __future__ import annotations
import secrets
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, Mapping


class ClassLabel(str, Enum):
    HEALTHY = "healthy"
    DIABETES = "diabetes"
    DYSLIPIDEMIA = "dyslipidemia"
    HYPERTENSION = "hypertension"
    OBESITY = "obesity"
    CV_RISK = "cardiovascular_risk"


class Sex(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class ActivityLevel(str, Enum):
    SEDENTARY = "Sedentary"
    MODERATE = "Moderate"
    HIGH = "High"


class SmokingStatus(str, Enum):
    NEVER = "Never"
    FORMER = "Former"
    CURRENT = "Current"


class AlcoholConsumption(str, Enum):
    NONE = "None"
    MODERATE = "Moderate"
    EXCESSIVE = "Excessive"


class DietQuality(str, Enum):
    POOR = "Poor"
    AVERAGE = "Average"
    GOOD = "Good"


@dataclass
class Patient:
    patient_id: str

    age: float
    sex: Sex
    height: float
    weight: float
    bmi: float

    heart_rate: float
    sbp: float
    dbp: float
    resp_rate: float
    temp: float

    fasting_glucose: float
    hba1c: float
    total_chol: float
    ldl: float
    hdl: float
    triglycerides: float

    physical_activity: ActivityLevel
    smoking: SmokingStatus
    alcohol: AlcoholConsumption
    diet_quality: DietQuality

    class_label: ClassLabel

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, Enum):
                d[key] = value.value
        return d

    def to_flat_values(self) -> dict[str, Any]:
        return self.to_dict()

    @staticmethod
    def new_id() -> str:
        return f"P{secrets.token_hex(5).upper()}"

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Patient":
        return cls(
            patient_id=str(data.get("patient_id") or cls.new_id()),
            age=float(data["age"]),
            sex=Sex(data["sex"]),
            height=float(data["height"]),
            weight=float(data["weight"]),
            bmi=float(data["bmi"]),
            heart_rate=float(data["heart_rate"]),
            sbp=float(data["sbp"]),
            dbp=float(data["dbp"]),
            resp_rate=float(data["resp_rate"]),
            temp=float(data["temp"]),
            fasting_glucose=float(data["fasting_glucose"]),
            hba1c=float(data["hba1c"]),
            total_chol=float(data["total_chol"]),
            ldl=float(data["ldl"]),
            hdl=float(data["hdl"]),
            triglycerides=float(data["triglycerides"]),
            physical_activity=ActivityLevel(data["physical_activity"]),
            smoking=SmokingStatus(data["smoking"]),
            alcohol=AlcoholConsumption(data["alcohol"]),
            diet_quality=DietQuality(data["diet_quality"]),
            class_label=ClassLabel(data["class_label"]),
        )


CONTINUOUS_FIELDS: tuple[str, ...] = (
    "age", "height", "weight", "bmi",
    "heart_rate", "sbp", "dbp", "resp_rate", "temp",
    "fasting_glucose", "hba1c", "total_chol", "ldl", "hdl", "triglycerides",
)

CATEGORICAL_FIELDS: tuple[str, ...] = (
    "sex", "physical_activity", "smoking", "alcohol", "diet_quality",
)
