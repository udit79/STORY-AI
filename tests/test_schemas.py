from app.schemas.page import (
    BoundingBox,
    TextRegion,
    PageRepresentation,
)


def test_text_region():
    region = TextRegion(
        id="t1",
        bbox=BoundingBox(
            x1=10,
            y1=20,
            x2=100,
            y2=60,
        ),
        raw_text="Hello!",
        confidence=0.95,
    )

    assert region.raw_text == "Hello!"
    assert region.confidence == 0.95


def test_page_representation():
    page = PageRepresentation(
        page_index=0,
        image_path="page1.png",
    )

    assert page.page_index == 0
    assert page.text_regions == []