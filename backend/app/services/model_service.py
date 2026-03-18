from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pandas as pd

from app.core.config import settings
from app.db.models import ModelRecordORM
from app.db.session import get_session
from app.schemas.common import ModelInfo
from app.schemas.train import TrainRequest

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor


class ModelService:
    """Manages training artifacts, model metadata, and inference loading."""

    def __init__(self) -> None:
        self._cache: dict[str, "TabularPredictor"] = {}

    def train(self, request: TrainRequest) -> ModelInfo:
        from autogluon.tabular import TabularPredictor

        dataset_path = Path(request.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        data = pd.read_csv(dataset_path)
        if request.target_column not in data.columns:
            raise ValueError(f"Target column '{request.target_column}' not found in dataset")

        model_id = str(uuid4())
        model_path = settings.models_root / model_id
        predictor = TabularPredictor(label=request.target_column, path=str(model_path)).fit(
            data,
            presets=request.presets,
            time_limit=request.time_limit,
        )

        info = ModelInfo(
            model_id=model_id,
            label_column=request.target_column,
            problem_type=str(predictor.problem_type),
            eval_metric=str(predictor.eval_metric),
            created_at=datetime.now(timezone.utc),
            model_path=str(model_path),
            source_dataset=str(dataset_path),
        )
        self._cache[model_id] = predictor
        self._upsert_db(info)
        self._write_registry(info)
        return info

    def predict(self, model_id: str, rows: list[dict[str, Any]]) -> list[Any]:
        if not rows:
            return []
        predictor = self._load_predictor(model_id)
        frame = pd.DataFrame(rows)
        predictions = predictor.predict(frame)
        return predictions.tolist()

    def get_model_info(self, model_id: str) -> ModelInfo:
        with get_session() as session:
            model_row = session.get(ModelRecordORM, model_id)
            if model_row is not None:
                return self._model_info_from_orm(model_row)

        model_record = settings.registry_root / f"{model_id}.json"
        if not model_record.exists():
            raise FileNotFoundError(f"Unknown model_id: {model_id}")

        payload = json.loads(model_record.read_text(encoding="utf-8"))
        return ModelInfo(**payload)

    def list_models(self) -> list[ModelInfo]:
        with get_session() as session:
            rows = session.query(ModelRecordORM).order_by(ModelRecordORM.created_at.desc()).all()
            if rows:
                return [self._model_info_from_orm(row) for row in rows]

        models: list[ModelInfo] = []
        for model_file in sorted(settings.registry_root.glob("*.json")):
            payload = json.loads(model_file.read_text(encoding="utf-8"))
            models.append(ModelInfo(**payload))
        return models

    def _load_predictor(self, model_id: str) -> TabularPredictor:
        from autogluon.tabular import TabularPredictor

        if model_id in self._cache:
            return self._cache[model_id]

        info = self.get_model_info(model_id)
        predictor = TabularPredictor.load(info.model_path)
        self._cache[model_id] = predictor
        return predictor

    @staticmethod
    def _write_registry(info: ModelInfo) -> None:
        record_path = settings.registry_root / f"{info.model_id}.json"
        record_path.write_text(info.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _upsert_db(info: ModelInfo) -> None:
        with get_session() as session:
            existing = session.get(ModelRecordORM, info.model_id)
            if existing is None:
                existing = ModelRecordORM(model_id=info.model_id)
                session.add(existing)

            existing.label_column = info.label_column
            existing.problem_type = info.problem_type
            existing.eval_metric = info.eval_metric
            existing.created_at = info.created_at
            existing.model_path = info.model_path
            existing.source_dataset = info.source_dataset
            session.commit()

    @staticmethod
    def _model_info_from_orm(row: ModelRecordORM) -> ModelInfo:
        return ModelInfo(
            model_id=row.model_id,
            label_column=row.label_column,
            problem_type=row.problem_type,
            eval_metric=row.eval_metric,
            created_at=row.created_at,
            model_path=row.model_path,
            source_dataset=row.source_dataset,
        )


model_service = ModelService()
