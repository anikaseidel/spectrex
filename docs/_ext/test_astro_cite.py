"""Standalone smoke test for the astronomy citation name formatting."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from astro_cite import AstroAuthorYearReferenceStyle
from pybtex.backends.plaintext import Backend
from pybtex.database import Entry, Person

backend = Backend()
style = AstroAuthorYearReferenceStyle()
person_style = style.person


def make_data(authors):
    e = Entry("article", fields={"year": "2020", "title": "T"})
    e.persons["author"] = [Person(a) for a in authors]
    return {"entry": e, "style": style}


cases = {
    "1 author": ["Smith, John"],
    "2 authors": ["Smith, John", "Jones, Alice"],
    "3 authors": ["Smith, John", "Jones, Alice", "Brown, Bob"],
    "4 authors": ["Smith, John", "Jones, Alice", "Brown, Bob", "Davis, Carol"],
}

print("Author names (short vs full):\n")
for label, authors in cases.items():
    data = make_data(authors)
    for full, tag in [(False, "short"), (True, "full")]:
        tpl = person_style.names("author", full=full)
        result = tpl.format_data(data)
        print(f"  {label} [{tag:5}]: {result.render(backend)}")
    print()
