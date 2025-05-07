from anomaly_reaper.infrastructure.database.models import ImageRecord, engine
from typing import Tuple, List, Optional
from sqlalchemy.orm import sessionmaker


def query_images(
    anomalies_only: bool = False,
    page: int = 1,
    page_size: int = 9,
    sort_by: str = "processed_at",
    sort_order: str = "desc",
) -> Tuple[List[ImageRecord], int, int]:
    """Query images from the database with pagination.

    Args:
        anomalies_only: If True, only return anomalous images
        page: The page number (1-based indexing)
        page_size: Number of records per page
        sort_by: Field to sort by
        sort_order: Sort direction ("asc" or "desc")

    Returns:
        Tuple of (List of ImageRecord objects, total count, total pages)
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        query = session.query(ImageRecord)

        if anomalies_only:
            query = query.filter(ImageRecord.is_anomaly == True)  # noqa: E712

        # Count total records for pagination
        total_count = query.count()

        # Calculate total pages
        total_pages = (
            (total_count + page_size - 1) // page_size if total_count > 0 else 1
        )

        # Apply sorting
        sort_column = getattr(ImageRecord, sort_by, ImageRecord.processed_at)
        if sort_order.lower() == "asc":
            query = query.order_by(sort_column.asc())
        else:
            query = query.order_by(sort_column.desc())

        # Apply pagination
        query = query.offset((page - 1) * page_size).limit(page_size)

        # Execute query
        results = query.all()

        return results, total_count, total_pages
    finally:
        session.close()


def query_image_by_id(image_id: str) -> Optional[ImageRecord]:
    """Query a single image by ID.

    Args:
        image_id: The unique identifier of the image

    Returns:
        ImageRecord object or None if not found
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        return session.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    finally:
        session.close()
