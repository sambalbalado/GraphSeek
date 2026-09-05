def test_package_can_be_imported() -> None:
    import graphseek

    assert graphseek.__name__ == "graphseek"
