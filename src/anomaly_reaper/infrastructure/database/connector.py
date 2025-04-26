from anomaly_reaper.infrastructure.database.models import ImageRecord, engine


def query_images(anomalies_only: bool = False) -> list[ImageRecord]:
    """Query images from the database.

    Args:
        anomalies_only: If True, only return anomalous images

    Returns:
        List of ImageRecord objects
    """
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        query = session.query(ImageRecord)

        if anomalies_only:
            query = query.filter(ImageRecord.is_anomaly == True)  # noqa: E712

        results = query.all()
        return results
    finally:
        session.close()
