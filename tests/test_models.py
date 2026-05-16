from app.models.paper import Paper


def test_paper_model_declares_expected_table_and_columns():
    columns = Paper.__table__.columns

    assert Paper.__tablename__ == "papers"
    assert set(columns.keys()) == {
        "id",
        "external_id",
        "title",
        "abstract",
        "keywords",
        "created_at",
        "last_updated",
    }
    assert columns["id"].primary_key is True
    assert columns["external_id"].unique is True
    assert columns["external_id"].nullable is False
    assert columns["title"].nullable is False
    assert columns["abstract"].nullable is False
    assert columns["keywords"].nullable is True
