# Konzept für den Chatbot "Data"

Team:

- Tobias Buess
- Yvo Keller
- Alexander Shanmugam

## Ziel

Ziel dieser Challenge ist die Entwicklung des Chatbots namens "Data". Er soll den Stundenten vom Studiengang Data Science zur Verfügung stehen, und ihnen Fragen rund um den Inhalt der Spaces zu den Modulen beanworten können, wobei er vorgegebenen ethischen Leitlinien folgen soll. Data soll auf die Inhalte der Modul-Spaces zugreifen und Standardanfragen mit Hilfe einer Wissensbasis beantworten können.

Der Bot soll auch Probleme des Benutzers erkennen und darauf moralisch adäquat reagieren, zum Beispiel mit aufmunternden Worten oder mit der Weitergabe an eine Ansprechperson. Er soll zudem zur Motivation der Studierenden beitragen.

Wir legen unseren Fokus in der Challenge darauf, eine Version des Bots zu bauen, die gut auf Deutsch funktioniert (Sprache der meisten Inhalte in der Wissensbasis). Dabei ist nicht ausgeschlossen, dass er auch auf Englisch funktioniert, aber wir werden uns nicht explizit darauf fokussieren.

## Fähigkeiten des Bots

Folgende Informationen soll der Bot auf Anfrage bereitstellen können:

- Seine Fähigkeiten erläutern
- Details zum Aufbau des Studiengangs (Konzept, Handbuch, Curriculum, Reglement)
- Details zum Modul
  - Fachexperten
  - Sprache
  - ECTS
  - Typ
  - Level
- Inhalte des Tabs "Porträt" der Module
- *optional*:
  - Lernmaterialien vorschlagen
  - Tab "Aufgaben"

Er soll NICHT:

- Auf Inhalte der Lernmaterialien zugreifen können (z.B. PDFs oder externe Links)

In seinem Verhalten soll der Bot folgende ethischen Leitlinien berücksichtigen:

- Motivierend, humorvoll und empathisch sein
- Auf Probleme des Users adäquat reagieren, wenn er diese erkennt (z.B. Stress im Studium, unzufriedenheit, depressive Phasen)und  Kontaktinformationen von Ansprechpersonen bereitstellen

## Wissensbasis

Die Wissensbasis des Bots soll auf mehreren Quellen aufbauen. Dies sind die zur Bereitstellung seiner Fähigkeiten notwendigen Informationen, einerseits aus dem Spaces DB Dump, andererseits aus den PDFs zum Studiengang (Konzept, Handbuch, Curriculum und Reglement).

Dabei soll der Bot bei der Beantwortung der Fragen den mitgeliefierten Kontext priorisieren. Wird eine Frage gestellt, die der Bot nicht auf Basis vom vorhandenen Kontext aus der Wissensbasis beantworten kann, deklariert er dies und greift zur Beantwortung der Anfrage entweder auf das Wissen im LLM zurück (z.B. bei "Was ist eine lineare Regression?"), oder lehnt die Beantwortung der Anfrage ab.

Die Inhalte können in Deutsch wie auch in Englisch vorhanden sein, was wir bei der Entwicklung berücksichtigen.

## Design

Wir gestalten einen Avator für "Data", der im Chat Interface angezeigt wird.

- Avatar Bild
- *optional*: Synthetische Stimme

## Architektur & Tech Stack

Der nachfolgenden Skizze kann die geplante Architektur entnommen werden. Die einzelnen Komponenten werden im Folgenden kurz erläutert, wobei Änderungen an der Architektur oder der eingesetzten Technologien im Verlauf der Challenge nicht ausgeschlossen sind.

![Architecture Sketch](architecture_sketch.jpeg)

### Chat Interface

Der Bot steht dem Nutzer in einem simplen Web Chat Interface zur Verfügung. Dieses wird mit Streamlit umgesetzt. Die Logik zur Verarbeitung von Anfragen durch den Bot bauen wir mit Python, wobei wir LangChain verwenden um mit den verschiedenen LLMs zu kommunizieren.

Tech Stack:

- Streamlit (Chat Interface)
- LangChain (Kommunikation mit LLM, Embeddings etc.)

### Prompt Classification (auch: npr MC1)

Der Bot unterscheidet an erster Stelle zwischen 3 Arten von Anfragen:

- question
- concern
- not a question or concern

Vorgehen:

- Fine-tunig eines BERT Modells zur Klassifikation der User Prompt

Tech Stack:

- HuggingFace Transformers library
- LLM: [BERT Base Multilingual](https://huggingface.co/bert-base-multilingual-cased) or similar

### Beantwortung von "concern"

Der Bot geht auf die Anliegen des Users ein, wenn er diese erkennt. Er soll dabei empathisch und motivierend reagieren, und dem User bei Bedarf Kontaktinformationen von Ansprechpersonen bereitstellen.

Vorgehen:

- Fine-tuning eines LLAMA2 Modells auf die ethischen Leitlinien für die Beratung und Unterstützung des Users im Bezug auf sein Anliegen.

Tech Stack:

- HuggingFace Transformers library
- LLM: [LLAMA2-13B-Chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) or similar

### Beantwortung von "question"

Der Bot beantwortet die Frage des Users, wenn er sie versteht und die Antwort in der Wissensbasis vorhanden ist. Ansonsten lehnt er die Beantwortung der Frage ab, oder weist den User darauf hin, dass in der Wissensbasis dazu nichts vorhanden ist, und versucht mit internem LLM Wissen weiter zu helfen.

Vorgehen:

- Chunking und Embedding des Kontexts in der Wissensbasis.
- Fine-tuning/Instruction-tuning eines LLAMA2 Modells auf die Beantwortung der Fragen aus gegebenem Kontext.

Tech Stack:

- Embeddings (BERT/Open AI)
- Vector Storage [PGVector](https://github.com/pgvector/pgvector)
- HuggingFace Transformers library
- LLM: [LLAMA2-13B-Chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) or similar

## Zielgruppe

Die Zielgruppe des Bots sind die Studierenden des Studiengangs Data Science. Wir haben zwei Personas definiert, die die Zielgruppe repräsentieren.

### Persona 1: Anna, die eifrige Studentin

#### Demografische Daten

- Alter: 23
- Geschlecht: Weiblich
- Beruflicher Hintergrund: Lehre als Informatikerin

#### Persönlichkeit

- Ehrgeizig und fokussiert
- Detailorientiert
- Liebt es, frühzeitig zu planen

#### Bedürfnisse und Ziele

- Will den besten Überblick über ihre Module haben
- Sucht immer nach zusätzlichen Ressourcen für bessere Lernerfolge
- Möchte auf dem Laufenden bleiben, was Änderungen im Curriculum betrifft

#### Nutzungsszenarien

- Fragt den Bot nach den Leistungsnachweisen in spezifischen Modulen
- Will wissen, welche Fachexperten für ein Modul zuständig sind
- Plant das nächste Semester und fragt den Bot nach nach Modulen im Curriculum

### Persona 2: Markus, der berufstätige Student

#### Demografische Daten

- Alter: 29
- Geschlecht: Männlich
- Beruflicher Hintergrund: Arbeitet Teilzeit im Rechnungswesen

#### Persönlichkeit

- Pragmatisch und zielorientiert
- Legt Wert auf Work-Life-Study-Balance
- Etwas stressanfällig aufgrund der vielen Verpflichtungen

#### Bedürfnisse und Ziele

- Sucht nach einem effizienten Weg, die Studieninformationen zu konsultieren
- Will möglichst wenig Zeit mit der Suche nach grundlegenden Informationen verbringen
- Sucht nach einer schnellen Möglichkeit, seine Fragen zu klären, um sich auf seine Arbeit und das Studium zu konzentrieren

#### Nutzungsszenarien

- Will schnell wissen, wie viele ECTS ein Modul hat
- Möchte erfahren, was ihn in einem spezifischen Modul erwartet
- Nutzt die motivierenden und empathischen Funktionen des Bots, um Stress abzubauen

Diese Personas können als Grundlage für die Entwicklung des Chatbots "Data" dienen. Sie repräsentieren die Bedürfnisse und Ziele der Zielgruppe und können dazu beitragen, die Funktionalität und das Verhalten des Bots optimal auszurichten.

## Privatshpäre

Der Datenschutz muss immer gewährleistet werden. Persönliche Daten sollen nicht an Drittanbieter wie OpenAI weitergegeben werden.
Das erreichen wir mit folgenden Massnahmen:

- Wir nutzen ein Named Entity Recognition Modell, um persönliche Daten wie Namen, Organisationen, Orte, und E-Mail Adessen zu erkennen und durch Platzhalter zu ersetzen. (z.B. Name mit Variable {name} ersetzen)
- Das LLM wird über die vorhandenen Variabeln instruiert, und behandelt diese auch für Antworten Platzhalter. (z.B. "Hallo {name}, wie kann ich dir helfen?")
- Anschliessend können wir die Platzhalter in der generierten Antwort wieder durch die erkannten Entitäten ersetzen, und dem User diese anzeigen.

## Evaluierung

Ein Chatbot, der falsche Antworten gibt, oder nicht auf die Fragen des Benutzers eingeht, ist nicht hilfreich. Er soll deshalb auf verschiedene Arten evaluiert werden.

### Qualitativ

- Factuality

### Quantitativ

Metriken wie:

- Coverage
- Tiefe
- Genauigkeit

## Was wir NICHT erreichen wollen

- Mehrsprachigkeit unterstützen
- Mehrere LLMs in der Evaluation gegenüberstellen
