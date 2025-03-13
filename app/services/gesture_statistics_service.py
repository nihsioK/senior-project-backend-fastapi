from app.models.gesture_statistics import ActionStatistic
from app.dependencies import get_db


class ActionStatisticService:
    @staticmethod
    def process_action(db, camera_id: str, action: str):
        if not db:
            db = next(get_db())
        statistic = db.query(ActionStatistic).filter_by(camera_id=camera_id, action=action).first()
        if statistic:
            statistic.count += 1
        else:
            statistic = ActionStatistic(camera_id=camera_id, action=action, count=1)
            db.add(statistic)
        db.commit()
        return statistic

    @staticmethod
    def get_camera_statistics(db, camera_id: str):
        return db.query(ActionStatistic).filter(ActionStatistic.camera_id == camera_id).all()
