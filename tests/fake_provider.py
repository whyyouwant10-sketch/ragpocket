"""Small deterministic provider used only by the optional plugin tests."""


class FakeProvider:
    name = "fake"
    model = "counts-v1"
    dimensions = 2

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text):
        folded = text.casefold()
        return [float(folded.count("deployment")), float(folded.count("retention"))]


class InvalidProvider(FakeProvider):
    def embed_documents(self, texts):
        return [[1.0] for _ in texts]
