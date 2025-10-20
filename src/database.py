from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from datetime import datetime
from typing import List, Optional
import json
import base64
import cv2
import numpy as np

Base = declarative_base()


class EventLog(Base):
    __tablename__ = "event_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    state = Column(String, nullable=False)
    dogs_detected = Column(Integer, default=0)
    humans_detected = Column(Integer, default=0)
    duration_unsupervised_seconds = Column(Float, nullable=True)
    frame_snapshot = Column(LargeBinary, nullable=True)
    detections_json = Column(String, nullable=True)
    alert_triggered = Column(Boolean, default=False)


class Database:
    def __init__(self, database_url: str = "sqlite+aiosqlite:///doodie_duty.db"):
        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def log_event(
        self,
        state: str,
        dogs_detected: int,
        humans_detected: int,
        duration_unsupervised_seconds: Optional[float] = None,
        frame_snapshot: Optional[np.ndarray] = None,
        detections: Optional[list] = None,
        alert_triggered: bool = False
    ) -> int:
        async with self.async_session() as session:
            frame_data = None
            if frame_snapshot is not None:
                _, buffer = cv2.imencode('.jpg', frame_snapshot, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_data = buffer.tobytes()

            detections_json = None
            if detections:
                detections_dict = [
                    {
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox": d.bbox,
                        "timestamp": d.timestamp.isoformat()
                    }
                    for d in detections
                ]
                detections_json = json.dumps(detections_dict)

            event = EventLog(
                timestamp=datetime.utcnow(),
                state=state,
                dogs_detected=dogs_detected,
                humans_detected=humans_detected,
                duration_unsupervised_seconds=duration_unsupervised_seconds,
                frame_snapshot=frame_data,
                detections_json=detections_json,
                alert_triggered=alert_triggered
            )

            session.add(event)
            await session.commit()
            return event.id

    async def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        state_filter: Optional[str] = None,
        alerts_only: bool = False
    ) -> List[dict]:
        async with self.async_session() as session:
            from sqlalchemy import select

            query = select(EventLog)

            if start_time:
                query = query.where(EventLog.timestamp >= start_time)
            if end_time:
                query = query.where(EventLog.timestamp <= end_time)
            if state_filter:
                query = query.where(EventLog.state == state_filter)
            if alerts_only:
                query = query.where(EventLog.alert_triggered == True)

            query = query.order_by(EventLog.timestamp.desc())
            query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            events = result.scalars().all()

            return [
                {
                    "id": event.id,
                    "timestamp": event.timestamp.isoformat(),
                    "state": event.state,
                    "dogs_detected": event.dogs_detected,
                    "humans_detected": event.humans_detected,
                    "duration_unsupervised_seconds": event.duration_unsupervised_seconds,
                    "alert_triggered": event.alert_triggered,
                    "has_snapshot": event.frame_snapshot is not None
                }
                for event in events
            ]

    async def get_event_snapshot(self, event_id: int) -> Optional[str]:
        async with self.async_session() as session:
            from sqlalchemy import select

            query = select(EventLog).where(EventLog.id == event_id)
            result = await session.execute(query)
            event = result.scalar_one_or_none()

            if event and event.frame_snapshot:
                return base64.b64encode(event.frame_snapshot).decode('utf-8')
            return None

    async def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> dict:
        async with self.async_session() as session:
            from sqlalchemy import select, func

            query = select(
                func.count(EventLog.id).label("total_events"),
                func.sum(EventLog.alert_triggered).label("total_alerts"),
                func.avg(EventLog.duration_unsupervised_seconds).label("avg_unsupervised_duration"),
                func.max(EventLog.duration_unsupervised_seconds).label("max_unsupervised_duration")
            )

            if start_time:
                query = query.where(EventLog.timestamp >= start_time)
            if end_time:
                query = query.where(EventLog.timestamp <= end_time)

            result = await session.execute(query)
            stats = result.one()

            state_query = select(
                EventLog.state,
                func.count(EventLog.id).label("count")
            )
            if start_time:
                state_query = state_query.where(EventLog.timestamp >= start_time)
            if end_time:
                state_query = state_query.where(EventLog.timestamp <= end_time)

            state_query = state_query.group_by(EventLog.state)
            state_result = await session.execute(state_query)
            state_counts = {row.state: row.count for row in state_result}

            return {
                "total_events": stats.total_events or 0,
                "total_alerts": int(stats.total_alerts or 0),
                "avg_unsupervised_duration": float(stats.avg_unsupervised_duration or 0),
                "max_unsupervised_duration": float(stats.max_unsupervised_duration or 0),
                "state_counts": state_counts
            }

    async def cleanup_old_events(self, days_to_keep: int = 30) -> int:
        async with self.async_session() as session:
            from sqlalchemy import delete
            from datetime import timedelta

            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            query = delete(EventLog).where(EventLog.timestamp < cutoff_date)
            result = await session.execute(query)
            await session.commit()

            return result.rowcount