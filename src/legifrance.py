from __future__ import annotations

import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Final
from urllib.parse import urljoin, urlparse

import urllib3
from urllib3.util import Retry, Timeout

LEGIFRANCE_ORIGIN: Final = "https://www.legifrance.gouv.fr"
_ALLOWED_HOST: Final = "www.legifrance.gouv.fr"
_ARTICLE_LABEL_RE: Final = re.compile(
    r"^Article\s+([A-ZÀ-ÖØ-Ý]*\s*\d[0-9A-ZÀ-ÖØ-öø-ÿ.\-]*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CorpusSpec:
    key: str
    label: str
    description: str
    seed_url: str
    legal_scope: str


CORPORA: Final[dict[str, CorpusSpec]] = {
    "ip": CorpusSpec(
        key="ip",
        label="Code de la propriété intellectuelle",
        description="French intellectual-property law: copyright, neighbouring rights, patents, marks and related rules.",
        seed_url=f"{LEGIFRANCE_ORIGIN}/codes/texte_lc/LEGITEXT000006069414/",
        legal_scope="Code de la propriété intellectuelle",
    ),
    "competition": CorpusSpec(
        key="competition",
        label="Code de commerce — Livre IV",
        description="French competition law: pricing freedom, anticompetitive practices, concentrations and related rules.",
        seed_url=(
            f"{LEGIFRANCE_ORIGIN}/codes/section_lc/"
            "LEGITEXT000005634379/LEGISCTA000006113741/"
        ),
        legal_scope="Livre IV : De la liberté des prix et de la concurrence",
    ),
}


class LegifranceError(RuntimeError):
    """Raised when a bounded Légifrance collection request cannot be completed safely."""


@dataclass(frozen=True)
class ArticleLink:
    article: str
    url: str


@dataclass(frozen=True)
class LegalProvision:
    corpus: str
    corpus_label: str
    article: str
    url: str
    text: str

    @property
    def source_key(self) -> str:
        compact = re.sub(r"\s+", "", self.article).lower()
        return f"legifrance:{self.corpus}:{compact}"

    @property
    def source_text(self) -> str:
        # Retrieval timestamps are deliberately excluded so unchanged law remains hash-stable.
        return (
            "Source officielle: Légifrance\n"
            f"Corpus: {self.corpus_label}\n"
            f"Article: {self.article}\n"
            f"URL: {self.url}\n\n"
            f"{self.text.strip()}"
        )

    @property
    def provenance(self) -> dict[str, str]:
        return {
            "provider": "Légifrance / DILA",
            "corpus": self.corpus,
            "corpus_label": self.corpus_label,
            "article": self.article,
            "url": self.url,
            "retrieval_mode": "public_html",
        }


@dataclass(frozen=True)
class ScrapeBatch:
    corpus: str
    seed_url: str
    discovered: int
    selected: int
    provisions: tuple[LegalProvision, ...]
    errors: tuple[dict[str, str], ...]


class _ArticleLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        values = dict(attrs)
        self._href = values.get("href")
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return
        label = re.sub(r"\s+", " ", " ".join(self._text)).strip()
        self.links.append((self._href, label))
        self._href = None
        self._text = []


class _VisibleTextParser(HTMLParser):
    _BLOCK_TAGS = {"article", "div", "h1", "h2", "h3", "h4", "h5", "li", "main", "p", "section", "br"}
    _IGNORED_TAGS = {"script", "style", "svg", "noscript", "header", "footer", "nav"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._main_depth = 0
        self._saw_main = False
        self.body: list[str] = []
        self.main: list[str] = []

    def _append(self, value: str) -> None:
        if self._ignored_depth:
            return
        self.body.append(value)
        if self._main_depth:
            self.main.append(value)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if tag == "main":
            self._saw_main = True
            self._main_depth += 1
        if tag in self._BLOCK_TAGS:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if tag in self._BLOCK_TAGS:
            self._append("\n")
        if tag == "main":
            self._main_depth = max(0, self._main_depth - 1)

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._append(data)

    @property
    def preferred(self) -> str:
        if self._saw_main and len("".join(self.main)) > 100:
            return "".join(self.main)
        return "".join(self.body)


def _lines(value: str) -> list[str]:
    return [re.sub(r"\s+", " ", line).strip() for line in value.splitlines() if line.strip()]


def parse_article_links(html: str, *, base_url: str) -> list[ArticleLink]:
    parser = _ArticleLinkParser()
    parser.feed(html)
    seen: set[str] = set()
    result: list[ArticleLink] = []
    for href, label in parser.links:
        if not _ARTICLE_LABEL_RE.match(label):
            continue
        url = urljoin(base_url, href)
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != _ALLOWED_HOST or not parsed.path.startswith("/codes/article_lc/"):
            continue
        article = re.sub(r"\s+", " ", label.removeprefix("Article")).strip()
        key = article.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(ArticleLink(article=article, url=url))
    return result


_METADATA_PREFIXES: Final = (
    "version en vigueur",
    "modifié par",
    "modifie par",
    "créé par",
    "cree par",
    "abrogé par",
    "abroge par",
    "masquer les articles",
    "accéder à la version",
    "acceder a la version",
    "imprimer la page",
    "copier le texte",
    "voir les versions",
    "comparer les versions",
    "déplier/replier",
    "deplier/replier",
)


def parse_article_text(html: str, *, article: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(html)
    lines = _lines(parser.preferred)
    target = f"article {article}".casefold()
    indices = [index for index, line in enumerate(lines) if line.casefold() == target]
    start = indices[-1] + 1 if indices else 0

    body: list[str] = []
    for line in lines[start:]:
        folded = line.casefold()
        if body and (folded.startswith("code de ") or folded.startswith("retour au sommaire")):
            break
        if body and folded.startswith("article ") and folded != target:
            break
        if folded.startswith(_METADATA_PREFIXES):
            continue
        if folded in {"ouvrir", "réduire", "reduire", "fermer"}:
            continue
        body.append(line)

    text = "\n".join(body).strip()
    if len(text) < 40:
        raise LegifranceError(f"article {article} did not contain enough legal text")
    if len(text) > 200_000:
        raise LegifranceError(f"article {article} exceeded the legal-text safety bound")
    return text


class LegifranceClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 12.0,
        request_delay_seconds: float = 0.2,
        max_page_bytes: int = 12_000_000,
        user_agent: str = "VLE-LEGATO/0.5 (+https://github.com/joedimos/legal_norm_argumentation)",
    ) -> None:
        self.request_delay_seconds = request_delay_seconds
        self.max_page_bytes = max_page_bytes
        self.user_agent = user_agent
        self._timeout = Timeout(connect=timeout_seconds, read=timeout_seconds)
        self._retries = Retry(
            total=2,
            connect=2,
            read=2,
            status=2,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        self._http = urllib3.PoolManager(num_pools=2, cert_reqs="CERT_REQUIRED")

    @staticmethod
    def _validate_url(url: str, *, article_only: bool = False) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != _ALLOWED_HOST:
            raise LegifranceError("refusing non-Légifrance URL")
        if not parsed.path.startswith("/codes/"):
            raise LegifranceError("refusing URL outside the Légifrance codes namespace")
        if article_only and not parsed.path.startswith("/codes/article_lc/"):
            raise LegifranceError("refusing non-article URL discovered by collector")

    def _get(self, url: str, *, article_only: bool = False, redirects: int = 0) -> str:
        self._validate_url(url, article_only=article_only)
        if redirects > 3:
            raise LegifranceError("too many Légifrance redirects")
        response = self._http.request(
            "GET",
            url,
            headers={"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml"},
            timeout=self._timeout,
            retries=self._retries,
            redirect=False,
            preload_content=True,
        )
        if response.status in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            if not location:
                raise LegifranceError("Légifrance redirect omitted Location")
            target = urljoin(url, location)
            self._validate_url(target, article_only=article_only)
            return self._get(target, article_only=article_only, redirects=redirects + 1)
        if response.status != 200:
            raise LegifranceError(f"Légifrance returned HTTP {response.status}")
        media_type = response.headers.get("Content-Type", "").lower()
        if media_type and "html" not in media_type:
            raise LegifranceError("Légifrance returned a non-HTML response")
        data = response.data
        if len(data) > self.max_page_bytes:
            raise LegifranceError("Légifrance page exceeded configured size limit")
        charset_match = re.search(r"charset=([\w-]+)", media_type)
        charset = charset_match.group(1) if charset_match else "utf-8"
        try:
            return data.decode(charset, errors="replace")
        except LookupError as exc:
            raise LegifranceError("Légifrance declared an unsupported character set") from exc

    def fetch_corpus(self, corpus: str, *, limit: int = 20, offset: int = 0) -> ScrapeBatch:
        if corpus not in CORPORA:
            raise ValueError(f"unknown Légifrance corpus: {corpus}")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if not 0 <= offset <= 100_000:
            raise ValueError("offset must be between 0 and 100000")

        spec = CORPORA[corpus]
        seed_html = self._get(spec.seed_url)
        links = parse_article_links(seed_html, base_url=spec.seed_url)
        if not links:
            raise LegifranceError("no article links were discovered; Légifrance page structure may have changed")
        selected = links[offset : offset + limit]
        provisions: list[LegalProvision] = []
        errors: list[dict[str, str]] = []

        for index, link in enumerate(selected):
            if index and self.request_delay_seconds:
                time.sleep(self.request_delay_seconds)
            try:
                html = self._get(link.url, article_only=True)
                provisions.append(
                    LegalProvision(
                        corpus=spec.key,
                        corpus_label=spec.label,
                        article=link.article,
                        url=link.url,
                        text=parse_article_text(html, article=link.article),
                    )
                )
            except LegifranceError as exc:
                errors.append({"article": link.article, "url": link.url, "error": str(exc)})

        return ScrapeBatch(
            corpus=spec.key,
            seed_url=spec.seed_url,
            discovered=len(links),
            selected=len(selected),
            provisions=tuple(provisions),
            errors=tuple(errors),
        )
