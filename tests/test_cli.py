from app import main


def test_initialise_rag_starts_shared_service(monkeypatch):
    started = []

    class FakeService:
        def start(self):
            started.append(True)

    monkeypatch.setattr(
        main,
        "RAGService",
        FakeService,
    )

    service = main.initialise_rag()

    assert isinstance(service, FakeService)
    assert started == [True]


def test_cli_routes_questions_through_shared_service(
    monkeypatch,
    capsys,
):
    questions = []

    class FakeService:
        def query(self, question):
            questions.append(question)
            return {
                "question": question,
                "answer": "Employees receive 25 days.",
                "retrieval_rejected": False,
                "retrieved_count": 1,
                "sources": [
                    {
                        "file": "EmployeeHandbook.pdf",
                        "page": 4,
                        "chunk_id": 8,
                        "similarity": 0.666,
                    }
                ],
            }

    answers = iter(
        (
            "How much annual leave?",
            "exit",
        )
    )
    monkeypatch.setattr(
        main,
        "initialise_rag",
        lambda: FakeService(),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(answers),
    )

    main.run()

    output = capsys.readouterr().out
    assert questions == ["How much annual leave?"]
    assert "Employees receive 25 days." in output
    assert "EmployeeHandbook.pdf, page 4" in output
    assert "similarity: 0.666" in output
    assert "Goodbye." in output


def test_cli_uses_service_abstention(monkeypatch, capsys):
    class FakeService:
        def query(self, question):
            return {
                "question": question,
                "answer": "No supported answer.",
                "retrieval_rejected": True,
                "retrieved_count": 0,
                "sources": [],
            }

    answers = iter(("Unsupported question", "exit"))
    monkeypatch.setattr(
        main,
        "initialise_rag",
        lambda: FakeService(),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(answers),
    )

    main.run()

    output = capsys.readouterr().out
    assert "No supported answer." in output
    assert "--- Sources ---" not in output
